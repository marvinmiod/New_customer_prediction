?	Ԛ??h@Ԛ??h@!Ԛ??h@	??5?@??5?@!??5?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ԛ??h@d;?O????A'????i@Y???????*	?????l[@2F
Iterator::Model?D???J??!9?.x?F@)Q?|a??1I?n??\@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????o??!?g$D?;@)??q????1Tч?{?8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<?R??!9??>T?3@)??y?):??1?????90@:Preprocessing2U
Iterator::Model::ParallelMapV2???S㥋?!?; Ҝ(@)???S㥋?1?; Ҝ(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????߮?!?Sч?{K@)?I+?v?1s{w @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!n???+@)????Mbp?1n???+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!?]??A?@)ŏ1w-!o?1?]??A?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=?U?????!?-\??5@)HP?s?b?1&|?v?? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??5?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d;?O????d;?O????!d;?O????      ??!       "      ??!       *      ??!       2	'????i@'????i@!'????i@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????JCPU_ONLYY??5?@b Y      Y@q?????_R@"?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?73.4936% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 