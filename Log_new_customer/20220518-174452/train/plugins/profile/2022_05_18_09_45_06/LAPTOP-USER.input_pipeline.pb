	󫺌EG@󫺌EG@!󫺌EG@	Tq娞@Tq娞@!Tq娞@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$󫺌EG@?0?*??A瘮e坈] @Y?q瑡鄹?*	烫烫烫e@2F
Iterator::Model+曉	h??!雁籶滵@)%u???1拐跕@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat霶?呺??![?4@)aTR'爥??1U汰蜨?2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=
祝p=??!/D忥bM@)脞6罌?1?Y涼/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?稝傗菢?!	l贠柪+@)@a脫?1!z|"&@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?O?!nuF*7@)傗菢粬??1?Y涼?"@:Preprocessing2U
Iterator::Model::ParallelMapV2漓?<,詩?!籶=?@)漓?<,詩?1籶=?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??t?!⑶w? z@)n??t?1⑶w? z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_vOf?!U汰蜨砒?)_vOf?1U汰蜨砒?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Tq娞@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?0?*???0?*??!?0?*??      ??!       "      ??!       *      ??!       2	瘮e坈] @瘮e坈] @!瘮e坈] @:      ??!       B      ??!       J	?q瑡鄹??q瑡鄹?!?q瑡鄹?R      ??!       Z	?q瑡鄹??q瑡鄹?!?q瑡鄹?JCPU_ONLYYTq娞@b 