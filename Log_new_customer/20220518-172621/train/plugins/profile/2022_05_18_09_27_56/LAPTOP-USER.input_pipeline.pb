	??v??@??v??@!??v??@	?ѻ"??@?ѻ"??@!?ѻ"??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??v??@|a2U0??A??o_? @Y??o_??*	??????c@2F
Iterator::Models??A϶?!H??-?L@)??N@a??1u??b??G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU???N@??!?'Jv?7@)??W?2ġ?1n.?d??5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?D???J??!??[?/@)?0?*??1??8H??)@:Preprocessing2U
Iterator::Model::ParallelMapV2S?!?uq??!Nc~,g? @)S?!?uq??1Nc~,g? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????ױ?!?t7?"?E@)??_vOv?1?/S;4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	?^)?p?!?^aM??@)	?^)?p?1?^aM??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!Ә?8??)?????g?1Ә?8??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz6?>W??!߾8H?0@)????Mb`?1C=e??&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?ѻ"??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|a2U0??|a2U0??!|a2U0??      ??!       "      ??!       *      ??!       2	??o_? @??o_? @!??o_? @:      ??!       B      ??!       J	??o_????o_??!??o_??R      ??!       Z	??o_????o_??!??o_??JCPU_ONLYY?ѻ"??@b 