use std::{collections::HashMap, any::Any};

pub struct MixinHolder (HashMap<String, Box<dyn Any>>);

pub trait HasMixins {
    fn mixin(&self, key: String) -> Option<Box<dyn Any>>;
}
