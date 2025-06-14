// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// the Scorer trait is used to calculate the score of a token in a document
// in general, the score is calculated as:
// sum over all query_weight(query_token) * doc_weight(freq, doc_tokens)
pub trait Scorer: Send + Sync {
    fn query_weight(&self, token: &str) -> f32;
    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32;
    // calculate the contribution of the token in the document
    // token: the token to score
    // freq: the frequency of the token in the document
    // doc_tokens: the number of tokens in the document
    fn score(&self, token: &str, freq: u32, doc_tokens: u32) -> f32 {
        self.query_weight(token) * self.doc_weight(freq, doc_tokens)
    }
    fn merge(scorers: &[&Self]) -> Self;
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BM25Scorer {
    nqs: HashMap<String, usize>,
    num_docs: usize,
    num_tokens: usize,
    avgdl: f32,
}

impl BM25Scorer {
    pub fn new(nqs: HashMap<String, usize>, num_docs: usize, num_tokens: usize) -> Self {
        let avgdl = num_tokens as f32 / num_docs as f32;
        Self {
            nqs,
            num_docs,
            num_tokens,
            avgdl,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    // the number of documents that contain the token
    pub fn nq(&self, token: &str) -> usize {
        *self.nqs.get(token).unwrap_or(&1)
    }
}

impl Scorer for BM25Scorer {
    fn merge(scorers: &[&Self]) -> Self {
        let mut nqs = HashMap::new();
        let mut num_docs = 0;
        let mut num_tokens = 0;
        for scorer in scorers {
            for (token, nq) in scorer.nqs.iter() {
                *nqs.entry(token.clone()).or_insert(0) += nq;
            }
            num_docs += scorer.num_docs;
            num_tokens += scorer.num_tokens;
        }
        Self::new(nqs, num_docs, num_tokens)
    }

    fn query_weight(&self, token: &str) -> f32 {
        let nq = self.nq(token);
        if nq == 0 {
            return 0.0;
        }
        idf(nq, self.num_docs)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let freq = freq as f32;
        let doc_tokens = doc_tokens as f32;
        let doc_norm = K1 * (1.0 - B + B * doc_tokens / self.avgdl);
        (K1 + 1.0) * freq / (freq + doc_norm)
    }
}

#[inline]
pub fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
}
