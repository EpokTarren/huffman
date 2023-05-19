use bit_vec::BitVec;
use std::{collections::BinaryHeap, rc::Rc};

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
    Node(usize, Direction, Rc<Self>),
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

fn table_recursive(tree: &Tree<u8>) -> [Option<Rc<DirectionNode>>; 256] {
    match &tree.inner {
        TreeInner::Leaf(n) => {
            const INIT: Option<Rc<DirectionNode>> = None;
            let mut directions = [INIT; 256];
            directions[*n as usize] = Some(Rc::new(DirectionNode::Leaf));
            directions
        }

        TreeInner::Node(left, right) => {
            let mut left = table_recursive(&left);
            let right = table_recursive(&right);

            for xs in left.iter_mut() {
                if let Some(xs) = xs {
                    *xs = Rc::new(DirectionNode::Node(
                        xs.len() + 1,
                        Direction::Left,
                        xs.clone(),
                    ));
                }
            }

            for (i, xs) in right
                .into_iter()
                .enumerate()
                .filter_map(|(i, xs)| xs.map(|xs| (i, xs)))
            {
                left[i] = Some(Rc::new(DirectionNode::Node(xs.len(), Direction::Right, xs)));
            }

            left
        }
    }
}

fn table(tree: &Tree<u8>) -> [Vec<Direction>; 256] {
    const INIT: Vec<Direction> = Vec::new();
    let mut directions = [INIT; 256];
    let direction_lists = table_recursive(tree);

    for (dst, src) in directions.iter_mut().zip(direction_lists.into_iter()) {
        if let Some(xs) = src {
            *dst = (*xs).clone().collect();
        }
    }

    directions
}

pub fn tree(buf: &[u8]) -> Tree<u8> {
    let mut frequencies = [0usize; 256];
    for &b in buf {
        frequencies[b as usize] += 1;
    }

    let mut frequencies = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &freq)| freq != 0)
        .map(|(i, &freq)| Tree {
            freq,
            inner: TreeInner::Leaf(i as u8),
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
            (None, None) => {
                return Tree {
                    freq: 0,
                    inner: TreeInner::Leaf(0),
                }
            }
            (None, Some(_)) => unreachable!(),
        }
    }
}

impl Tree<u8> {
    fn encode(&self, w: &mut BitVec) {
        match &self.inner {
            TreeInner::Leaf(b) => {
                w.push(true);
                w.append(&mut BitVec::from_bytes(&[*b]));
            }

            TreeInner::Node(left, right) => {
                w.push(false);
                left.encode(w);
                right.encode(w);
            }
        }
    }
}

pub fn encode_tree(
    buf: &[u8],
    tree: &Tree<u8>,
    output: &mut impl std::io::Write,
) -> std::io::Result<usize> {
    assert!(!buf.is_empty(), "Buffer must be non empty");

    let table = table(&tree);
    let mut w = BitVec::from_bytes(&buf.len().to_le_bytes());
    tree.encode(&mut w);

    buf.iter()
        .flat_map(|&b| table[b as usize].iter())
        .for_each(|d| w.push(*d != Direction::Left));

    output.write_all(&w.to_bytes()).map(|_| 0)
}

pub fn encode(buf: &[u8], output: &mut impl std::io::Write) -> std::io::Result<usize> {
    assert!(!buf.is_empty(), "Buffer must be non empty");

    let tree = tree(buf);
    encode_tree(buf, &tree, output)
}

fn read_byte<R: Iterator<Item = bool>>(r: &mut R) -> Option<u8> {
    (0..8).try_fold(0, |acc, _| r.next().map(|n| (acc << 1) | n as u8))
}

impl Tree<u8> {
    fn decode<R: Iterator<Item = bool>>(r: &mut R) -> Self {
        if r.next().unwrap() == true {
            Self {
                freq: 0,
                inner: TreeInner::Leaf(read_byte(r).unwrap()),
            }
        } else {
            Self {
                freq: 0,
                inner: TreeInner::Node(Box::new(Self::decode(r)), Box::new(Self::decode(r))),
            }
        }
    }

    fn decode_byte<R: Iterator<Item = bool>>(&self, r: &mut R) -> u8 {
        let mut tree = self;
        loop {
            match &tree.inner {
                TreeInner::Leaf(b) => return *b,
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

pub fn decode(r: &[u8]) -> Vec<u8> {
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
