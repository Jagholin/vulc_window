use crate::logger::Logger;
use crate::vertex_type::VertexStruct;
use anyhow::{bail, Error};
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::data::mesh::{PolygonVertex, PolygonVertexIndex, PolygonVertices};
use fbxcel_dom::v7400::object::{geometry::TypedGeometryHandle, TypedObjectHandle};
use std::fs::File;
use std::io::{BufReader, Write};
use cgmath::Point3;
use cgmath::{prelude::*, Point2, Quaternion, Vector3};

// we know that T is some T: Iterator<Item=PolygonVertex>
struct PolygonIteratorImpl<T> {
    inner_iter: T,
}

impl<T> Iterator for PolygonIteratorImpl<T>
where
    T: Iterator<Item = PolygonVertex>,
{
    type Item = Vec<PolygonVertex>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut res: Self::Item = Vec::new();
        for t in self.inner_iter.by_ref() {
            if t.is_end() {
                res.push(t);
                break;
            }
            res.push(t);
        }
        if res.is_empty() {
            None
        } else {
            Some(res)
        }
    }
}

trait PolygonIterator {
    type RetType;
    fn to_polygons(self) -> PolygonIteratorImpl<Self::RetType>;
}
struct MappedPolyVertexIterator<T> {
    inner_iter: T,
}

impl<T> Iterator for MappedPolyVertexIterator<T>
where
    T: Iterator<Item = i32>,
{
    type Item = PolygonVertex;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next().map(PolygonVertex::new)
    }
}

impl<I> PolygonIterator for I
where
    // T: Iterator<Item=PolygonVertex>,
    I: Iterator<Item = i32>,
{
    type RetType = MappedPolyVertexIterator<I>;
    fn to_polygons(self) -> PolygonIteratorImpl<Self::RetType> {
        PolygonIteratorImpl {
            inner_iter: MappedPolyVertexIterator { inner_iter: self },
        }
    }
}

// needs T::Item to be Clone, otherwise can't hold previous items for next calls to next()
struct TriplesIterator<'a, T> {
    inner_iter: Option<Box<dyn Iterator<Item = T> + 'a>>, // Option just so we have Default value for mem::take
    prev_item: Option<T>,
    pprev_item: Option<T>,
    first_call: bool,
}

impl<'a, T> Iterator for TriplesIterator<'a, T>
where
    T: Clone + 'a,
{
    type Item = (T, T, T);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pprev_item.is_none() {
            self.pprev_item = self.inner_iter.as_mut().unwrap().next().clone();
        }
        if self.prev_item.is_none() {
            self.prev_item = self.inner_iter.as_mut().unwrap().next().clone();
        }
        let (pprev, prev) = (self.pprev_item.clone(), self.prev_item.clone());
        match (pprev, prev) {
            (None, _) | (_, None) => None,
            (Some(pprev), Some(prev)) => {
                let next_item = self.inner_iter.as_mut().unwrap().next();
                if let Some(next_item) = next_item {
                    if self.first_call {
                        let my_iter = std::mem::take(&mut self.inner_iter).unwrap();
                        let my_iter = Box::new(my_iter.chain([pprev.clone(), prev.clone()]));
                        self.inner_iter = Some(my_iter);
                        
                        self.first_call = false;
                    }
                    self.pprev_item = self.prev_item.clone();
                    self.prev_item = Some(next_item.clone());
                    Some((pprev.clone(), prev.clone(), next_item))
                } else {
                    None
                }
            }
        }
    }
}

trait IntoTriples {
    fn into_triples<'a>(self) -> TriplesIterator<'a, Self::Item>
    where
        Self: Iterator + Sized + 'a;
}

impl<T> IntoTriples for T
where
    T: Iterator,
{
    fn into_triples<'a>(self) -> TriplesIterator<'a, T::Item>
    where
        T: 'a,
    {
        let b = Box::new(self);
        TriplesIterator {
            inner_iter: Some(b),
            prev_item: None,
            pprev_item: None,
            first_call: true,
        }
    }
}

// calculates whether v is inside a triangle specified by (p1, p2, p3)
// by calculating its barycenter coordinates
fn inside_triangle(v: Point2<f64>, p1: Point2<f64>, p2: Point2<f64>, p3: Point2<f64>) -> bool {
    let df = -p1.y * p2.x + p1.x * p2.y + p1.y * p3.x - p2.y * p3.x - p1.x * p3.y + p2.x * p3.y;
    // barycentric coordinates of v
    let s = (p2.x * p3.y - p2.y * p3.x) / df + v.x * (p2.y - p3.y) / df + v.y * (p3.x - p2.x) / df;
    let u = (p1.y * p3.x - p1.x * p3.y) / df + v.x * (p3.y - p1.y) / df + v.y * (p1.x - p3.x) / df;
    let t = 1.0 - s - u;

    // println!("s {}, u {}, t {}", s, u, t);

    // if point is inside and determinant is positive, all signs will be +
    // if point is inside, but determinant is negative, all signs will be -
    let all_pos = s > 0.0 && u > 0.0 && t > 0.0;
    let all_neg = s < 0.0 && u < 0.0 && t < 0.0;

    all_pos || all_neg
}

fn ear_clipping(
    mut verts: Vec<(PolygonVertexIndex, Point2<f64>)>,
) -> Vec<[(PolygonVertexIndex, Point2<f64>); 3]> {
    let mut result = vec![];
    // let mut ears = vec![];
    // TODO: this can be optimized, but for now it's good enough(also it works)
    // we assume there aren't many ngons in our models, as they will mostly consisit of quads
    loop {
        let mut ear_index = None;
        for triple in verts.iter().copied().into_triples() {
            let ((ix1, p1), ear @ (ix, p2), (ix2, p3)) = triple;
            let ab = p2 - p1;
            let bc = p3 - p2;
            let angle = ab.angle(bc);
            if angle.0 < 0.0 {
                // we have a reflex vertex here, cant be an ear
                continue;
            }
            if verts
                .iter()
                .filter(|(index, _)| *index != ix1 && *index != ix && *index != ix2 )
                .any(|(_, v)| inside_triangle(v.clone(), p1, p2, p3))
            {
                // not an ear
                continue;
            }
            result.push([triple.0, triple.1, triple.2]);
            ear_index = Some(ix);
            break;
        }
        // Its OK to panic, because if it is None, there is a problem with the algorithm
        let ear_index = ear_index.expect("Couldnt find an ear for some reason");
        verts.retain(|(i, _)| *i != ear_index);
        if verts.len() == 3 {
            break;
        }
    }
    // add the last triangle
    result.push([verts[0], verts[1], verts[2]]);
    result
}

fn polygon_winding(verts: impl Iterator<Item = Point2<f64>>) -> cgmath::Rad<f64> {
    let mut result = cgmath::Rad(0.0);
    for (p1, p2, p3) in verts.into_triples() {
        let ab = p2 - p1;
        let bc = p3 - p2;
        result += ab.angle(bc);
    }
    result
}

fn triangulate_ngon(
    verts: &PolygonVertices,
    pvis: &[PolygonVertexIndex],
    out: &mut Vec<[PolygonVertexIndex; 3]>,
) -> Result<(), Error> {
    let mut points = Vec::new();
    for ba in pvis {
        let v = verts.control_point(ba).unwrap();
        let point: Point3<f64> = cgmath::point3(v.x, v.y, v.z);
        points.push(IndexedPoint {
            point,
            index: ba.to_owned(),
        });
    }
    if points.len() == 3 {
        // already a triangle
        out.push([pvis[0], pvis[1], pvis[2]]);
        return Ok(());
    }
    if points.len() == 4 {
        let mut candidates = Vec::new();
        // 2 possible triangulations:
        // 0, 1, 2 and 2, 3, 0
        // 1, 2, 3 and 3, 0, 1
        // try both and take one that gives less area
        let tri_1 = [0, 1, 2];
        let tri_2 = [2, 3, 0];
        candidates.push((tri_1, tri_2));
        let tri_1 = [1, 2, 3];
        let tri_2 = [3, 0, 1];
        candidates.push((tri_1, tri_2));
        let minimal_tri = candidates
            .iter()
            .min_by(|c1, c2| {
                let area_lhs = cross_area(c1.0.map(|x| points[x].point))
                    + cross_area(c1.1.map(|x| points[x].point));
                let area_rhs = cross_area(c2.0.map(|x| points[x].point))
                    + cross_area(c2.1.map(|x| points[x].point));
                area_lhs.partial_cmp(&area_rhs).unwrap()
            })
            .unwrap();
        out.push(minimal_tri.0.map(|x| points[x].index));
        out.push(minimal_tri.1.map(|x| points[x].index));
    } else {

        // 1. find th normal by making cross product
        let mut norm = Vector3::new(0.0, 0.0, 0.0);
        for (p1, p2, p3) in points.iter().into_triples() {
            let ab = p2.point - p1.point;
            let bc = p3.point - p2.point;
            norm += ab.cross(bc);
        }
        let norm = norm.normalize();

        // 2. Create orthonormal basis by finding such rotation R that Rn = z
        let rot = Quaternion::between_vectors(Vector3::unit_z(), norm);
        let axis_i = rot.rotate_vector(Vector3::unit_x());
        let axis_j = rot.rotate_vector(Vector3::unit_y());
        // Make sure that {i,j,k} is indeed an orthonormal basis in R3
        debug_assert!(axis_i.dot(norm).abs() < 0.002);
        debug_assert!(axis_j.dot(norm).abs() < 0.002);
        debug_assert!(axis_i.dot(axis_j).abs() < 0.002);
        debug_assert!((axis_i.magnitude2() - 1.0).abs() < 0.002);
        debug_assert!((axis_j.magnitude2() - 1.0).abs() < 0.002);
        
        // 3. Create points projected into 2d plane given by {i, j}
        let projected_points: Vec<_> = points
            .iter()
            .map(|p| {
                (
                    p.index,
                    Point2::new(p.point.dot(axis_i), p.point.dot(axis_j)),
                )
            })
            .collect();
        let clipped = ear_clipping(projected_points);
        for ps in clipped.into_iter() {
            out.push([ps[0].0, ps[1].0, ps[2].0]);
        }
    }
    Ok(())
}

struct IndexedPoint<I, T> {
    point: T,
    index: I,
}

fn cross_area<T: cgmath::BaseNum + cgmath::num_traits::Float>(tris: [Point3<T>; 3]) -> T {
    let ab = tris[1] - tris[0];
    let ac = tris[2] - tris[0];
    ab.cross(ac).magnitude()
}

pub fn read_fbx_document(
    file_name: &str,
    logger: &mut Logger<impl Write>,
) -> Result<Vec<VertexStruct>, Error> {
    let file = File::open(file_name).expect("Cant open fbx file");
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    match AnyDocument::from_seekable_reader(reader).expect("Failed to load document") {
        AnyDocument::V7400(fbxver, doc) => {
            println!("FBX version {:#?}", fbxver.major_minor());
            println!("objects count: {}", doc.objects().count());
            logger.log(format!("{:#?}", doc.tree().debug_tree()).as_str());
            for obj in doc.objects() {
                println!(
                    "New object: {}, class {}, sub {}",
                    obj.name().unwrap_or_default(),
                    obj.class(),
                    obj.subclass()
                );
                if let TypedObjectHandle::Geometry(TypedGeometryHandle::Mesh(mesh)) =
                    obj.get_typed()
                {
                    // iterate over polygons
                    let vertices = mesh.polygon_vertices().unwrap();
                    let triangulated_verts = vertices.triangulate_each(triangulate_ngon).unwrap();

                    // let mut counter = 1;
                    let normals_data = mesh
                        .layers()
                        .find_map(|l| {
                            for entry in l.layer_element_entries() {
                                if let Ok(TypedLayerElementHandle::Normal(h)) =
                                    entry.typed_layer_element()
                                {
                                    return Some(h.normals().unwrap());
                                }
                            }
                            None
                        })
                        .unwrap();
                    for cpi in triangulated_verts.triangle_vertex_indices() {
                        let point = triangulated_verts.control_point(cpi).unwrap();
                        let normal = normals_data.normal(&triangulated_verts, cpi).unwrap();
                        result.push(VertexStruct {
                            position: [point.x as f32, point.y as f32, point.z as f32],
                            normal: [normal.x as f32, normal.y as f32, normal.z as f32],
                        });
                        // println!("{}: point: {:?}, normal: {:?}", counter, point, normal);
                        // counter += 1;
                    }
                }
            }
            // println!("tree: {:?}", );
            Ok(result)
        }
        _ => {
            bail!("Unexpected FBX version");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use cgmath::{InnerSpace, Point2, Vector2};

    use super::{inside_triangle, IntoTriples};

    #[test]
    fn empty_iterator() {
        let iter_arr: [usize; 0] = [];
        let mut iter = iter_arr.iter().into_triples();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iterate_small_array() {
        let iter_arr: [usize; 1] = [1];
        let mut iter = iter_arr.iter().into_triples();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        let iter_arr: [usize; 2] = [1, 2];
        let mut iter = iter_arr.iter().into_triples();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn single_element() {
        let iter_arr: [usize; 3] = [1, 2, 3];
        let mut iter = iter_arr.iter().into_triples();
        assert_eq!(iter.next(), Some((&1, &2, &3)));
        assert_eq!(iter.next(), Some((&2, &3, &1)));
        assert_eq!(iter.next(), Some((&3, &1, &2)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn more_elements() {
        let iter_arr: [usize; 5] = [1, 2, 3, 4, 5];
        let mut iter = iter_arr.iter().into_triples();
        assert_eq!(iter.next(), Some((&1, &2, &3)));
        assert_eq!(iter.next(), Some((&2, &3, &4)));
        assert_eq!(iter.next(), Some((&3, &4, &5)));
        assert_eq!(iter.next(), Some((&4, &5, &1)));
        assert_eq!(iter.next(), Some((&5, &1, &2)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn angle_test() {
        let p1: Vector2<f64> = Vector2::new(1.0, -1.0);
        let p2: Vector2<f64> = Vector2::new(1.0, 1.0);
        let p3: Vector2<f64> = Vector2::new(2.0, 2.0);
        let ab = p2 - p1;
        let bc = p3 - p2;
        assert_eq!(ab.angle(bc).0, -PI / 4.0)
    }

    #[test]
    fn strange_triangle() {
        let p1 = Point2::new(0.4434109926223753, -0.21228843927383423);
        let p2 = Point2::new(0.7315716743469237, 0.061153948307037354);
        let p3 = Point2::new(0.8538796901702879, 0.061153948307037354);
        let p4 = Point2::new(0.988878011703491, 0.21228837966918945);
        assert!(!inside_triangle(p1, p2, p3, p4));
    }
}
