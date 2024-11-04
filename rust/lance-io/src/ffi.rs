// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchReader, UInt64Array};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use lance_core::{Result, ROW_ID};

use crate::stream::RecordBatchStream;

#[pin_project::pin_project]
struct RecordBatchIteratorAdaptor<S: RecordBatchStream> {
    schema: SchemaRef,

    #[pin]
    stream: S,

    handle: tokio::runtime::Handle,
}

impl<S: RecordBatchStream> RecordBatchIteratorAdaptor<S> {
    fn new(stream: S, schema: SchemaRef, handle: tokio::runtime::Handle) -> Self {
        Self {
            schema,
            stream,
            handle,
        }
    }
}

impl<S: RecordBatchStream + Unpin> arrow::record_batch::RecordBatchReader
    for RecordBatchIteratorAdaptor<S>
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl<S: RecordBatchStream + Unpin> Iterator for RecordBatchIteratorAdaptor<S> {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.handle
            .block_on(async { self.stream.next().await })
            .map(|r| r.map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}

/// Wrap a [`RecordBatchStream`] into an [FFI_ArrowArrayStream].
pub fn to_ffi_arrow_array_stream(
    stream: impl RecordBatchStream + std::marker::Unpin + 'static,
    handle: tokio::runtime::Handle,
) -> Result<FFI_ArrowArrayStream> {
    let schema = stream.schema();
    let arrow_stream = RecordBatchIteratorAdaptor::new(stream, schema, handle);
    let reader = FFI_ArrowArrayStream::new(Box::new(arrow_stream));

    Ok(reader)
}

/// Wrap a [`RecordBatchStream`] into an [FFI_ArrowArrayStream] for jni call
/// transformer the _rowid UInt64 to Int64 since java only has Long type
pub fn to_ffi_jni_arrow_array_stream(
    stream: impl RecordBatchStream + std::marker::Unpin + 'static,
    handle: tokio::runtime::Handle,
) -> Result<FFI_ArrowArrayStream> {
    let schema = stream.schema();
    let arrow_stream = JniRecordBatchIteratorAdaptor::new(stream, schema, handle);
    let reader = FFI_ArrowArrayStream::new(Box::new(arrow_stream));

    Ok(reader)
}

#[pin_project::pin_project]
struct JniRecordBatchIteratorAdaptor<S: RecordBatchStream> {
    schema: SchemaRef,
    #[pin]
    stream: S,
    handle: tokio::runtime::Handle,
}
impl<S: RecordBatchStream> JniRecordBatchIteratorAdaptor<S> {
    fn new(stream: S, schema: SchemaRef, handle: tokio::runtime::Handle) -> Self {
        Self {
            schema,
            stream,
            handle,
        }
    }
}
impl<S: RecordBatchStream + Unpin> arrow::record_batch::RecordBatchReader
for JniRecordBatchIteratorAdaptor<S>
{
    fn schema(&self) -> SchemaRef {
        let mut new_fields = Vec::new();
        for field in self.schema.clone().fields() {
            if field.name() == ROW_ID {
                let new_field = match field.data_type() {
                    DataType::UInt64 => Field::new(field.name().clone(), DataType::Int64, field.is_nullable()),
                    // Add more conversions as needed
                    _ => field.as_ref().clone() // Keep the original if no conversion is needed
                };
                new_fields.push(new_field);
            } else {
                new_fields.push(field.as_ref().clone());
            }
        }
        Arc::new(Schema::new(new_fields))
    }
}
impl<S: RecordBatchStream + Unpin> Iterator for JniRecordBatchIteratorAdaptor<S> {
    type Item = std::result::Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.handle
            .block_on(async { self.stream.next().await })
            .map(|r| r.map(|batch| {
                match batch.schema().index_of(ROW_ID) {
                    Ok(index) => {
                        let mut new_columns = batch.columns().to_vec();
                        let uint64_array = batch.column(index).as_any().downcast_ref::<UInt64Array>().unwrap();
                        // Create a new Int64Array
                        let mut int_values: Vec<i64> = Vec::with_capacity(uint64_array.len());
                        for i in 0..uint64_array.len() {
                            // Convert UInt64 to Int64 (you may want to handle overflow here)
                            int_values.push(uint64_array.value(i) as i64);
                        }
                        let int_array = Int64Array::from(int_values);
                        new_columns[index] = Arc::new(int_array.clone()); // Replace the specified column
                        // Create a new RecordBatch with the updated columns
                        RecordBatch::try_new(self.schema(), new_columns).unwrap()
                    }
                    Err(_err) => batch
                }
            }).map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}
