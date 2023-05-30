use bit_vec::BitVec;
use std::collections::{BinaryHeap, HashMap};

pub trait Alphabet: Sized + Eq + std::hash::Hash {
    type Map: IntoIterator<Item = (Self, usize)>;

    fn init_map() -> Self::Map;
    fn increment_map(&self, map: &mut Self::Map);
    fn from_bits(it: &mut impl Iterator<Item = bool>) -> Option<Self>;
    fn write(&self, bits: &mut BitVec);
}

impl Alphabet for u8 {
    type Map = [(u8, usize); 256];

    fn init_map() -> Self::Map {
        [(0, 0); 256]
    }

    fn increment_map(&self, map: &mut Self::Map) {
        map[*self as usize].0 = *self;
        map[*self as usize].1 += 1;
    }

    fn from_bits(it: &mut impl Iterator<Item = bool>) -> Option<Self> {
        read_byte(it)
    }

    fn write(&self, bits: &mut BitVec) {
        bits.extend(BitVec::from_bytes(&[*self]))
    }
}

impl<const N: usize> Alphabet for [u8; N] {
    type Map = HashMap<Self, usize>;

    fn init_map() -> Self::Map {
        Self::Map::new()
    }

    fn increment_map(&self, map: &mut Self::Map) {
        map.entry(*self).and_modify(|c| *c += 1).or_insert(1);
    }

    fn from_bits(it: &mut impl Iterator<Item = bool>) -> Option<Self> {
        let mut res = [0; N];
        for b in res.iter_mut() {
            *b = read_byte(it)?;
        }

        return Some(res);
    }

    fn write(&self, bits: &mut BitVec) {
        bits.extend(BitVec::from_bytes(self))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Tree<T> {
    pub freq: usize,
    pub inner: TreeInner<T>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TreeInner<T> {
    Leaf(T),
    Node(Box<Tree<T>>, Box<Tree<T>>),
}

impl<T: Eq> Ord for Tree<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.freq.cmp(&self.freq)
    }
}

impl<T: Eq> PartialOrd for Tree<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Left = 0,
    Right = 1,
}

#[derive(Debug, Clone)]
enum DirectionNode {
    Leaf,
    Node(usize, Direction, Box<Self>),
}

impl Iterator for DirectionNode {
    type Item = Direction;

    fn next(&mut self) -> Option<Direction> {
        match self {
            Self::Node(_, direction, next) => {
                let direction = direction.clone();

                match next.as_ref() {
                    Self::Leaf => *self = Self::Leaf,
                    Self::Node(i, d, n) => *self = Self::Node(*i, d.clone(), n.clone()),
                }

                Some(direction)
            }
            Self::Leaf => None,
        }
    }
}

impl ExactSizeIterator for DirectionNode {
    fn len(&self) -> usize {
        match self {
            Self::Node(i, _, _) => *i,
            Self::Leaf => 0,
        }
    }
}

fn table_recursive<A: Alphabet + Clone>(tree: &Tree<A>) -> HashMap<A, Box<DirectionNode>> {
    match &tree.inner {
        TreeInner::Leaf(n) => {
            let mut directions = HashMap::new();
            directions.insert(n.clone(), Box::new(DirectionNode::Leaf));
            return directions;
        }

        TreeInner::Node(left, right) => {
            let mut left = table_recursive(&left);
            let right = table_recursive(&right);

            for (_, xs) in left.iter_mut() {
                *xs = Box::new(DirectionNode::Node(
                    xs.len() + 1,
                    Direction::Left,
                    xs.clone(),
                ));
            }

            for (n, xs) in right.into_iter() {
                let direction = Box::new(DirectionNode::Node(xs.len() + 1, Direction::Right, xs));
                left.insert(n, direction);
            }

            left
        }
    }
}

fn table<A: Alphabet + Clone>(tree: &Tree<A>) -> HashMap<A, Vec<Direction>> {
    table_recursive(tree)
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()
}

pub fn tree<A: Alphabet>(buf: &[A]) -> Tree<A> {
    let mut frequencies = A::init_map();
    for b in buf {
        b.increment_map(&mut frequencies);
    }

    let mut frequencies = frequencies
        .into_iter()
        .filter(|(_, freq)| *freq != 0)
        .map(|(ch, freq)| Tree {
            freq,
            inner: TreeInner::Leaf(ch),
        })
        .collect::<BinaryHeap<_>>();

    loop {
        match (frequencies.pop(), frequencies.pop()) {
            (Some(left), Some(right)) => {
                let freq = left.freq + right.freq;
                frequencies.push(Tree {
                    freq,
                    inner: TreeInner::Node(Box::new(left), Box::new(right)),
                });
            }
            (Some(tree), None) => return tree,
            (None, None) | (None, Some(_)) => unreachable!(),
        }
    }
}

impl<A: Alphabet> Tree<A> {
    fn encode(&self, w: &mut BitVec) {
        match &self.inner {
            TreeInner::Leaf(b) => {
                w.push(true);
                A::write(b, w);
            }

            TreeInner::Node(left, right) => {
                w.push(false);
                left.encode(w);
                right.encode(w);
            }
        }
    }
}

pub fn encode_tree<A: Alphabet + Clone>(
    buf: &[A],
    tree: &Tree<A>,
    output: &mut impl std::io::Write,
) -> std::io::Result<usize> {
    assert!(!buf.is_empty(), "Buffer must be non empty");

    let table = table(&tree);
    let mut w = BitVec::from_bytes(&buf.len().to_le_bytes());
    tree.encode(&mut w);

    buf.iter()
        .flat_map(|b| table.get(b).unwrap())
        .for_each(|d| w.push(*d != Direction::Left));

    output.write_all(&w.to_bytes()).map(|_| 0)
}

pub fn encode<A: Alphabet + Clone>(
    buf: &[A],
    output: &mut impl std::io::Write,
) -> std::io::Result<usize> {
    assert!(!buf.is_empty(), "Buffer must be non empty");

    let tree = tree(buf);
    encode_tree(buf, &tree, output)
}

pub fn read_byte<R: Iterator<Item = bool>>(r: &mut R) -> Option<u8> {
    (0..8).try_fold(0, |acc, _| r.next().map(|n| (acc << 1) | n as u8))
}

impl<A: Alphabet + Clone> Tree<A> {
    fn decode<R: Iterator<Item = bool>>(r: &mut R) -> Self {
        if r.next().unwrap() == true {
            Self {
                freq: 0,
                inner: TreeInner::Leaf(A::from_bits(r).unwrap()),
            }
        } else {
            Self {
                freq: 0,
                inner: TreeInner::Node(Box::new(Self::decode(r)), Box::new(Self::decode(r))),
            }
        }
    }

    fn decode_byte<R: Iterator<Item = bool>>(&self, r: &mut R) -> A {
        let mut tree = self;
        loop {
            match &tree.inner {
                TreeInner::Leaf(b) => return b.clone(),
                TreeInner::Node(left, right) => {
                    tree = if matches!(r.next(), Some(false)) {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }
}

pub fn decode<A: Alphabet + Clone>(r: &[u8]) -> Vec<A> {
    assert!(r.len() >= 10, "Must fit header information");

    let len = usize::from_le_bytes(r[..8].try_into().expect("slice to have length at least 10"));
    let bits = BitVec::from_bytes(&r[8..]);
    let mut iter = bits.iter();
    let tree = Tree::decode(&mut iter);

    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(tree.decode_byte(&mut iter));
    }

    vec
}
