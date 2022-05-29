use anyhow::{Error, bail};
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::data::mesh::{PolygonVertex, PolygonVertices, PolygonVertexIndex};
use fbxcel_dom::v7400::object::{TypedObjectHandle, geometry::TypedGeometryHandle};
use std::fs::File;
use std::io::{BufReader, Write};
use std::ops::Sub;
use crate::logger::Logger;
use crate::vertex_type::VertexStruct;

// lets extract polygon data from vertex index stream
trait Vector3 {
    type ScalarType;
    fn from_v(x: Self::ScalarType, y: Self::ScalarType, z: Self::ScalarType) -> Self
        where Self: Sized;
    fn cross_product(&self, rhs: &Self) -> Self
        where Self: Sized;
    fn dot_product(&self, rhs: &Self) -> Self::ScalarType
        where Self: Sized;
    fn x(&self) -> Self::ScalarType;
    fn y(&self) -> Self::ScalarType;
    fn z(&self) -> Self::ScalarType;
    fn diff(&self, rhs: &Self) -> Self
    where Self: Sized, Self::ScalarType: Sub<Output = Self::ScalarType> {
        Self::from_v(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl Vector3 for (f32, f32, f32) {
    type ScalarType = f32;
    fn cross_product(&self, rhs: &Self) -> Self {
        (self.1*rhs.2 - self.2*rhs.1, self.2*rhs.0 - self.0*rhs.2, self.0*rhs.1 - self.1*rhs.0)
    }
    fn dot_product(&self, rhs: &Self) -> Self::ScalarType {
        self.0*rhs.0 + self.1*rhs.1 + self.2*rhs.2
    }
    fn x(&self) -> Self::ScalarType {
        self.0
    }
    fn y(&self) -> Self::ScalarType {
        self.1
    }
    fn z(&self) -> Self::ScalarType {
        self.2
    }
    fn from_v(x: Self::ScalarType, y: Self::ScalarType, z: Self::ScalarType) -> Self {
        (x, y, z)
    }
}

impl Vector3 for [f32; 3] {
    type ScalarType = f32;
    fn cross_product(&self, rhs: &Self) -> Self {
        [
           self[1] * rhs[2] - self[2] * rhs[1],
           self[2] * rhs[0] - self[0] * rhs[2],
           self[0] * rhs[1] - self[1] * rhs[0]
        ]
    }
    fn dot_product(&self, rhs: &Self) -> Self::ScalarType {
        self.iter().zip(rhs.iter()).map(|(x, y)| x*y).fold(0.0, |acc, val| acc + val)
    }
    fn x(&self) -> Self::ScalarType {
        self[0]
    }
    fn y(&self) -> Self::ScalarType {
        self[1]
    }
    fn z(&self) -> Self::ScalarType {
        self[2]
    }
    fn from_v(x: Self::ScalarType, y: Self::ScalarType, z: Self::ScalarType) -> Self {
        [x, y, z]
    }
}

fn vector_len(vec: [f32; 3]) -> f32 {
    vec.dot_product(&vec).sqrt()
}

// we know that T is some T: Iterator<Item=PolygonVertex>
struct PolygonIteratorImpl<T> {
    inner_iter: T,
}

impl<T> Iterator for PolygonIteratorImpl<T>
where T: Iterator<Item=PolygonVertex> {
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
where T: Iterator<Item=i32> {
    type Item = PolygonVertex; 
    fn next(&mut self) -> Option<Self::Item> {
        self.inner_iter.next().map(PolygonVertex::new)
    }
}

impl<I> PolygonIterator for I
where // T: Iterator<Item=PolygonVertex>,
I: Iterator<Item=i32> {

    type RetType = MappedPolyVertexIterator<I>;
    fn to_polygons(self) -> PolygonIteratorImpl<Self::RetType> {
        PolygonIteratorImpl {
            inner_iter: MappedPolyVertexIterator{
                inner_iter: self
            }
        }
    }
}

fn triangulate_quads(verts: &PolygonVertices, pvis: &[PolygonVertexIndex], out: &mut Vec<[PolygonVertexIndex; 3]>) -> Result<(), Error> {
    let mut points = Vec::new();
    for ba in pvis {
        let v = verts.control_point(ba).unwrap();
        let point = [v.x as f32, v.y as f32, v.z as f32];
        points.push(IndexedPoint{point, index: ba.to_owned()});
    }
    if points.len() == 3 {
        // already a triangle
        out.push([pvis[0], pvis[1], pvis[2]]);
        return Ok (());
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
        let minimal_tri = candidates.iter().min_by(|c1, c2| {
            let area_lhs = cross_area(c1.0.map(|x| points[x].point)) + triangle_area(c1.1.map(|x| points[x].point));
            let area_rhs = cross_area(c2.0.map(|x| points[x].point)) + triangle_area(c2.1.map(|x| points[x].point));
            area_lhs.partial_cmp(&area_rhs).unwrap()
        }).unwrap();
        out.push(minimal_tri.0.map(|x| points[x].index));
        out.push(minimal_tri.1.map(|x| points[x].index));
    }
    else {
        // return Err(TriangulateError{}.into());
        bail!("Triangulation of polygons with #verts > 4 is not supported yet");
    }
    Ok(())
}

struct IndexedPoint<I, T>{
    point: T,
    index: I,
}

impl<I,T> IndexedPoint<I, T> {
    fn get_index(&self) -> &I {
        &self.index
    }
    fn get_point(&self) -> &T {
        &self.point
    }
}

fn triangle_area (tris: [[f32; 3]; 3]) -> f32 {
    let a = tris[0];
    let b = tris[1];
    let c = tris[2];
    let ab: Vec<f32> = a.into_iter().zip(b.into_iter()).map(|(x,y)| y - x).collect();
    let ac: Vec<f32> = a.into_iter().zip(c.into_iter()).map(|(x,y)| y - x).collect();
    let len_ab = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
    let len_ac = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
    let cos_phi = (ab[0] * ac[0] + ab[1] * ac[1] + ab[2] * ac[2]) / (len_ab * len_ac);
    let sin_phi = (1.0 - cos_phi*cos_phi).sqrt();
    0.5 * len_ab * len_ac * sin_phi
}

fn cross_area (tris: [[f32; 3]; 3]) -> f32{
    let ab = tris[1].diff(&tris[0]);
    let ac = tris[2].diff(&tris[0]);
    vector_len(ab.cross_product(&ac))
}

pub fn read_fbx_document(file_name: &str, logger: &mut Logger<impl Write>) -> Result<Vec<VertexStruct>, Error> {
    let file = File::open(file_name).expect("Cant open fbx file");
    let reader = BufReader::new(file);
    let mut result = Vec::new();

    match AnyDocument::from_seekable_reader(reader).expect("Failed to load document") {
        AnyDocument::V7400(fbxver, doc) => {
            println!("FBX version {:#?}", fbxver.major_minor());
            println!("objects count: {}", doc.objects().count());
            logger.log(format!("{:#?}", doc.tree().debug_tree()).as_str());
            for obj in doc.objects() {
                println!("New object: {}, class {}, sub {}", obj.name().unwrap_or_default(), obj.class(), obj.subclass());
                if let TypedObjectHandle::Geometry(TypedGeometryHandle::Mesh(mesh)) = obj.get_typed() {
                    // iterate over polygons
                    let vertices = mesh.polygon_vertices().unwrap();
                    let triangulated_verts = vertices.triangulate_each(triangulate_quads).unwrap();

                    let mut counter = 1;
                    let normals_data = mesh.layers().find_map(|l| {
                        for entry in l.layer_element_entries() {
                            if let Ok(TypedLayerElementHandle::Normal(h)) = entry.typed_layer_element() {
                                return Some(h.normals().unwrap());
                            }
                        }
                        None
                    }).unwrap();
                    for cpi in triangulated_verts.triangle_vertex_indices() {
                        let point = triangulated_verts.control_point(cpi).unwrap();
                        let normal = normals_data.normal(&triangulated_verts, cpi).unwrap();
                        result.push(VertexStruct { position: [point.x as f32, point.y as f32, point.z as f32], normal: [normal.x as f32, normal.y as f32, normal.z as f32] });
                        println!("{}: point: {:?}, normal: {:?}", counter, point, normal);
                        counter += 1;
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
