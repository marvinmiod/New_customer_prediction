	?HP????HP???!?HP???	:tT??@:tT??@!:tT??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?HP???jM??S??A8gDio???Y??????*	     ?c@2F
Iterator::Model??ǘ????!$I?$I?D@)?	h"lx??1??i??i@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?-????!?(??(?>@)46<?R??1뺮뺮;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??ܵ?|??!?q?q4@)p_?Q??1EQEQ0@:Preprocessing2U
Iterator::Model::ParallelMapV2???<,Ԋ?!)??(?? @)???<,Ԋ?1)??(?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??V?/???!۶m۶mM@)?j+??݃?1)??(??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9??v??z?! ? ?@)9??v??z?1 ? ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!?m۶m?@);?O??nr?1?m۶m?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??B?iޡ?!??(??(6@)??_vOf?1ܶm۶m??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no99tT??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	jM??S??jM??S??!jM??S??      ??!       "      ??!       *      ??!       2	8gDio???8gDio???!8gDio???:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????JCPU_ONLYY9tT??@b 