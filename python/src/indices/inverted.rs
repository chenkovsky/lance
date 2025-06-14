use std::sync::Arc;

use lance_index::scalar::inverted::query::collect_tokens;
use lance_index::scalar::inverted::query::FtsSearchParams;
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use lance_index::scalar::inverted::{BM25Scorer, InvertedPartitionMetadata, Scorer};
use lance_index::scalar::{inverted::InvertedIndex, ScalarIndex};
use pyo3::{PyObject, PyRef, PyResult};
use std::collections::{HashMap, HashSet};

use crate::RT;

#[pyclass]
#[derive(Clone)]
pub struct PyInvertedIndex {
    column: String,
    uuid: String,
    inner: Arc<dyn ScalarIndex>,
}

impl PyInvertedIndex {
    pub fn new(column: String, uuid: String, index: Arc<dyn ScalarIndex>) -> Self {
        Self {
            column,
            uuid,
            inner: index,
        }
    }

    fn as_inverted_index(&self) -> &InvertedIndex {
        self.inner.as_any().downcast_ref::<InvertedIndex>().unwrap()
    }
}

#[pymethods]
impl PyInvertedIndex {
    pub fn __repr__(&self) -> String {
        format!("PyInvertedIndex({:?}, {:?})", self.column, self.uuid)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn uuid(&self) -> String {
        self.uuid.clone()
    }

    pub fn column(&self) -> String {
        self.column.clone()
    }

    pub fn partitions(self_: PyRef<'_, Self>) -> PyResult<Vec<PyInvertedPartitionMetadata>> {
        Ok(self_
            .as_inverted_index()
            .partitions()
            .iter()
            .map(|p| PyInvertedPartitionMetadata::new(p.clone()))
            .collect())
    }

    #[pyo3(signature = (query, inclusive=None))]
    pub fn tokenize(
        self_: PyRef<'_, Self>,
        query: &str,
        inclusive: Option<HashSet<String>>,
    ) -> PyResult<HashSet<String>> {
        let mut tokenizer = self_.as_inverted_index().tokenizer().clone();
        let query_tokens = collect_tokens(query, &mut tokenizer, inclusive.as_ref())
            .into_iter()
            .collect::<HashSet<_>>();
        Ok(query_tokens)
    }

    #[pyo3(signature = (tokens, params=None, partition_ids=None))]
    pub fn fuzzy_nq(
        self_: PyRef<'_, Self>,
        tokens: HashSet<String>,
        params: Option<Bound<'_, PyDict>>,
        partition_ids: Option<Vec<u64>>,
    ) -> PyResult<HashMap<String, usize>> {
        let fts_params = params
            .map(|params| {
                let max_expansions = params
                    .get_item("max_expansions")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?;
                let fuzziness = params
                    .get_item("fuzziness")?
                    .map(|v| v.extract::<Option<u32>>())
                    .transpose()?;
                let prefix_length = params
                    .get_item("prefix_length")?
                    .map(|v| v.extract::<u32>())
                    .transpose()?;

                Ok::<_, PyErr>(FtsSearchParams {
                    max_expansions: max_expansions.unwrap_or(50),
                    fuzziness: fuzziness.unwrap_or(Some(0)),
                    prefix_length: prefix_length.unwrap_or(0),
                    ..Default::default()
                })
            })
            .transpose()?
            .unwrap_or_else(FtsSearchParams::new);

        RT.runtime
            .block_on(self_.as_inverted_index().fuzzy_nq(
                &tokens,
                &fts_params,
                partition_ids.as_ref(),
            ))
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyInvertedPartitionMetadata {
    inner: Arc<InvertedPartitionMetadata>,
}

impl PyInvertedPartitionMetadata {
    pub fn new(inner: Arc<InvertedPartitionMetadata>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInvertedPartitionMetadata {
    pub fn __repr__(&self) -> String {
        format!("PyInvertedPartitionMetadata({:?})", self.inner.id())
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn id(&self) -> u64 {
        self.inner.id()
    }

    pub fn num_tokens(&self) -> usize {
        self.inner.num_tokens()
    }

    pub fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    pub fn fragments(&self) -> Vec<u32> {
        self.inner.fragments().iter().cloned().collect()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyBM25Scorer {
    pub(crate) inner: Arc<BM25Scorer>,
}

impl PyBM25Scorer {
    pub fn new(inner: Arc<BM25Scorer>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBM25Scorer {
    pub fn __repr__(&self) -> String {
        format!("PyBM25Scorer({:?})", self.inner)
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn score(&self, token: &str, freq: u32, doc_tokens: u32) -> f32 {
        self.inner.score(token, freq, doc_tokens)
    }

    #[staticmethod]
    pub fn merge(scorers: Vec<Self>) -> PyResult<Self> {
        let scorer =
            BM25Scorer::merge(&scorers.iter().map(|s| s.inner.as_ref()).collect::<Vec<_>>());
        Ok(Self::new(Arc::new(scorer)))
    }

    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let scorer = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!("Could not load BM25Scorer due to error: {}", err))
        })?;
        Ok(Self::new(Arc::new(scorer)))
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.as_ref()).map_err(|err| {
            PyValueError::new_err(format!("Could not dump BM25Scorer due to error: {}", err))
        })
    }

    pub fn __reduce__(&self, py: Python) -> PyResult<(PyObject, PyObject)> {
        let state = self.to_json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.index.bm25")?
            .getattr("BM25Scorer")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }
}
