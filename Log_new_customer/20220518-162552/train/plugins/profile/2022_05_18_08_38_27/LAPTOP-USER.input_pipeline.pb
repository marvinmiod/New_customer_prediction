	??@???@??@???@!??@???@	P{)M?+??P{)M?+??!P{)M?+??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??@???@?V?/?'??AI.?!??@Y'???????*	43333?`@2F
Iterator::Model"?uq??!M?]?/wG@)Ș?????1????C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???????!1 K?9@)???Q???1?/7Āt6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???߾??!???L?4@)
ףp=
??1?c)?`?0@:Preprocessing2U
Iterator::Model::ParallelMapV2??0?*??!?5?'?!@)??0?*??1?5?'?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?QI??&??!?}?zЈJ@)9??v??z?1A:?2	v@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen??t?!?c)?`W@)n??t?1?c)?`W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!CH?R&	@)?J?4q?1CH?R&	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?W[?????!?;?Ӛ6@)Ǻ???f?1?]?/7? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9O{)M?+??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V?/?'???V?/?'??!?V?/?'??      ??!       "      ??!       *      ??!       2	I.?!??@I.?!??@!I.?!??@:      ??!       B      ??!       J	'???????'???????!'???????R      ??!       Z	'???????'???????!'???????JCPU_ONLYYO{)M?+??b 