	?_vO? @?_vO? @!?_vO? @	?aQ??4@?aQ??4@!?aQ??4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_vO? @?]K?=??A??C?l??Y/?$???*	     p`@2F
Iterator::Model1?Zd??!???ĚWD@)??|гY??1?,???@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatf??a?֤?!?Q??X?>@)HP?sע?1?{7A?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???߾??!?p?$?4@)??ͪ?Ֆ?1??X?J?0@:Preprocessing2U
Iterator::Model::ParallelMapV2??0?*??!?0Bd_?!@)??0?*??1?0Bd_?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+??????!q]`;e?M@)???_vO~?1?R=?n?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??ZӼ?t?!???M?@)??ZӼ?t?1???M?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q????o?!(?????@)?q????o?1(?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapu????!q?$ּ6@){?G?zd?1u8?~k??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?aQ??4@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]K?=???]K?=??!?]K?=??      ??!       "      ??!       *      ??!       2	??C?l????C?l??!??C?l??:      ??!       B      ??!       J	/?$???/?$???!/?$???R      ??!       Z	/?$???/?$???!/?$???JCPU_ONLYY?aQ??4@b 