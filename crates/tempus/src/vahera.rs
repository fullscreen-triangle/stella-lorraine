//! # vaHera Abstract Syntax Tree
//!
//! The four frozen variants of the vaHera AST (stability contract: no variant
//! is ever added, removed, or renamed).
//!
//! Every Tempus trigger program compiles to a `Fragment`.  The kernel's
//! executor evaluates `Fragment`s to `Value`s; the PVE type-checks them
//! before dispatch.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// Named operation identifier (frozen string key into the operation registry).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationName(pub String);

impl OperationName {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for OperationName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Value — the runtime result space
// ---------------------------------------------------------------------------

/// Runtime value produced by fragment evaluation.
///
/// Mirrors the vaHera `Value` inductive type from the formal spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Record(BTreeMap<String, Value>),
    /// Partition label carried as a Value (used by detector subsystem operations).
    Label(crate::partition::PartitionLabel),
    /// Accept / reject outcome (L1 trigger decision).
    Decision(TriggerDecision),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerDecision {
    Accept,
    Reject,
}

impl Value {
    pub fn as_num(&self) -> Option<f64> {
        match self { Value::Num(v) => Some(*v), _ => None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Value::Bool(b) => Some(*b), _ => None }
    }
    pub fn as_label(&self) -> Option<&crate::partition::PartitionLabel> {
        match self { Value::Label(l) => Some(l), _ => None }
    }
    pub fn as_decision(&self) -> Option<TriggerDecision> {
        match self { Value::Decision(d) => Some(*d), _ => None }
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null       => "Null",
            Value::Bool(_)    => "Bool",
            Value::Num(_)     => "Num",
            Value::Str(_)     => "Str",
            Value::List(_)    => "List",
            Value::Record(_)  => "Record",
            Value::Label(_)   => "Label",
            Value::Decision(_)=> "Decision",
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment — the four frozen AST variants
// ---------------------------------------------------------------------------

/// vaHera abstract syntax tree.
///
/// The four variants are **frozen**: downstream consumers rely on their
/// permanence (stability contract from the formal spec).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Fragment {
    /// A runtime value embedded directly in the AST.
    /// Corresponds to `Literal(v)` in the formal spec.
    Literal(Value),

    /// An invocation of a named operation with named arguments.
    /// Corresponds to `Call(Op, args̄)` in the formal spec.
    Call {
        op: OperationName,
        /// Named-argument map (BTreeMap for deterministic ordering).
        args: BTreeMap<String, Fragment>,
    },

    /// Sequential composition with carry-threading.
    /// Each stage receives the output of the previous stage as its implicit
    /// `input` argument.
    /// Corresponds to `Compose(F̄)` in the formal spec.
    Compose(Vec<Fragment>),

    /// A typed placeholder awaiting substitution.
    /// Corresponds to `Hole(x)` in the formal spec.
    Hole(String),
}

impl Fragment {
    // --- Constructors -------------------------------------------------------

    pub fn literal(v: impl Into<Value>) -> Self {
        Self::Literal(v.into())
    }

    pub fn call(op: impl Into<String>, args: BTreeMap<String, Fragment>) -> Self {
        Self::Call { op: OperationName::new(op), args }
    }

    pub fn call0(op: impl Into<String>) -> Self {
        Self::Call { op: OperationName::new(op), args: BTreeMap::new() }
    }

    pub fn call1(op: impl Into<String>, key: impl Into<String>, arg: Fragment) -> Self {
        let mut args = BTreeMap::new();
        args.insert(key.into(), arg);
        Self::Call { op: OperationName::new(op), args }
    }

    pub fn compose(stages: Vec<Fragment>) -> Self {
        Self::Compose(stages)
    }

    pub fn hole(name: impl Into<String>) -> Self {
        Self::Hole(name.into())
    }

    // --- Queries ------------------------------------------------------------

    pub fn is_literal(&self) -> bool { matches!(self, Self::Literal(_)) }
    pub fn is_call(&self)    -> bool { matches!(self, Self::Call { .. }) }
    pub fn is_compose(&self) -> bool { matches!(self, Self::Compose(_)) }
    pub fn is_hole(&self)    -> bool { matches!(self, Self::Hole(_)) }

    /// Returns `true` if the fragment contains no unresolved `Hole` variants.
    pub fn is_ground(&self) -> bool {
        match self {
            Self::Literal(_)          => true,
            Self::Hole(_)             => false,
            Self::Call { args, .. }   => args.values().all(Self::is_ground),
            Self::Compose(stages)     => stages.iter().all(Self::is_ground),
        }
    }

    /// Syntactic size (number of AST nodes).
    pub fn size(&self) -> usize {
        match self {
            Self::Literal(_) | Self::Hole(_) => 1,
            Self::Call { args, .. } => 1 + args.values().map(Self::size).sum::<usize>(),
            Self::Compose(stages)   => 1 + stages.iter().map(Self::size).sum::<usize>(),
        }
    }

    /// Depth of the deepest Compose chain (0 for leaf nodes).
    pub fn compose_depth(&self) -> usize {
        match self {
            Self::Literal(_) | Self::Hole(_) => 0,
            Self::Call { args, .. } => args.values().map(Self::compose_depth).max().unwrap_or(0),
            Self::Compose(stages) => {
                let inner = stages.iter().map(Self::compose_depth).max().unwrap_or(0);
                inner + 1
            }
        }
    }

    /// Substitute a `Hole` by name with a concrete fragment.
    pub fn substitute(self, name: &str, replacement: &Fragment) -> Self {
        match self {
            Self::Hole(ref n) if n == name => replacement.clone(),
            Self::Hole(_) | Self::Literal(_) => self,
            Self::Call { op, args } => Self::Call {
                op,
                args: args.into_iter()
                    .map(|(k, v)| (k, v.substitute(name, replacement)))
                    .collect(),
            },
            Self::Compose(stages) => Self::Compose(
                stages.into_iter().map(|s| s.substitute(name, replacement)).collect()
            ),
        }
    }
}

// Convenience conversions into Value
impl From<f64>   for Value { fn from(v: f64)   -> Self { Value::Num(v) } }
impl From<bool>  for Value { fn from(v: bool)  -> Self { Value::Bool(v) } }
impl From<&str>  for Value { fn from(v: &str)  -> Self { Value::Str(v.to_string()) } }
impl From<String>for Value { fn from(v: String)-> Self { Value::Str(v) } }
impl From<crate::partition::PartitionLabel> for Value {
    fn from(l: crate::partition::PartitionLabel) -> Self { Value::Label(l) }
}
impl From<TriggerDecision> for Value {
    fn from(d: TriggerDecision) -> Self { Value::Decision(d) }
}

// ---------------------------------------------------------------------------
// Evaluation semantics (bare executor, no subsystems)
// ---------------------------------------------------------------------------

/// A registered provider: takes named `Value` arguments, returns a `Value`.
pub type ProviderFn = Box<dyn Fn(BTreeMap<String, Value>) -> crate::Result<Value> + Send + Sync>;

/// Operation registry: maps `OperationName` → provider function.
pub struct OperationRegistry {
    ops: std::collections::HashMap<String, ProviderFn>,
}

impl OperationRegistry {
    pub fn new() -> Self { Self { ops: Default::default() } }

    pub fn register(&mut self, name: impl Into<String>, f: ProviderFn) {
        self.ops.insert(name.into(), f);
    }

    pub fn contains(&self, name: &str) -> bool { self.ops.contains_key(name) }
    pub fn names(&self) -> impl Iterator<Item = &str> { self.ops.keys().map(String::as_str) }

    /// Invoke a registered operation.
    pub fn invoke(
        &self,
        op: &OperationName,
        args: BTreeMap<String, Value>,
    ) -> crate::Result<Value> {
        let f = self.ops.get(op.as_str()).ok_or_else(|| crate::TempusError::UnregisteredOperation {
            op: op.0.clone(),
        })?;
        f(args)
    }
}

impl Default for OperationRegistry {
    fn default() -> Self { Self::new() }
}

/// Bare evaluation of a `Fragment` against an `OperationRegistry`.
///
/// This is the executor `eval` function from the formal spec.
/// Returns `None` if the fragment contains an unresolved `Hole`.
pub fn eval(fragment: &Fragment, registry: &OperationRegistry) -> crate::Result<Option<Value>> {
    match fragment {
        Fragment::Literal(v) => Ok(Some(v.clone())),

        Fragment::Hole(_) => Ok(None),

        Fragment::Call { op, args } => {
            // Evaluate all arguments first.
            let mut resolved: BTreeMap<String, Value> = BTreeMap::new();
            for (key, frag) in args {
                match eval(frag, registry)? {
                    Some(v) => { resolved.insert(key.clone(), v); }
                    None    => return Ok(None), // unresolved hole propagates upward
                }
            }
            Ok(Some(registry.invoke(op, resolved)?))
        }

        Fragment::Compose(stages) => {
            if stages.is_empty() { return Ok(None); }
            let mut carry: Value = match eval(&stages[0], registry)? {
                Some(v) => v,
                None    => return Ok(None),
            };
            // Thread carry as the implicit `input` argument for each subsequent stage.
            for stage in &stages[1..] {
                let mut stage_with_carry = stage.clone();
                stage_with_carry = stage_with_carry.substitute("input", &Fragment::Literal(carry));
                carry = match eval(&stage_with_carry, registry)? {
                    Some(v) => v,
                    None    => return Ok(None),
                };
            }
            Ok(Some(carry))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_evaluates_to_itself() {
        let reg = OperationRegistry::new();
        let f = Fragment::literal(42.0_f64);
        assert_eq!(eval(&f, &reg).unwrap(), Some(Value::Num(42.0)));
    }

    #[test]
    fn hole_returns_none() {
        let reg = OperationRegistry::new();
        let f = Fragment::hole("x");
        assert_eq!(eval(&f, &reg).unwrap(), None);
    }

    #[test]
    fn compose_threads_carry() {
        let mut reg = OperationRegistry::new();
        // Op that doubles its `input` number.
        reg.register("double", Box::new(|args| {
            let v = args["input"].as_num().unwrap();
            Ok(Value::Num(v * 2.0))
        }));
        let f = Fragment::compose(vec![
            Fragment::literal(3.0_f64),
            Fragment::call0("double"),
            Fragment::call0("double"),
        ]);
        assert_eq!(eval(&f, &reg).unwrap(), Some(Value::Num(12.0)));
    }

    #[test]
    fn fragment_size_and_depth() {
        let f = Fragment::compose(vec![
            Fragment::literal(1.0_f64),
            Fragment::call1("op", "x", Fragment::literal(2.0_f64)),
        ]);
        assert_eq!(f.size(), 4);       // Compose + Lit + Call + Lit
        assert_eq!(f.compose_depth(), 1);
    }

    #[test]
    fn substitute_replaces_hole() {
        let f = Fragment::compose(vec![
            Fragment::hole("x"),
            Fragment::call0("op"),
        ]);
        let g = f.substitute("x", &Fragment::literal(99.0_f64));
        assert!(g.is_ground());
    }
}
