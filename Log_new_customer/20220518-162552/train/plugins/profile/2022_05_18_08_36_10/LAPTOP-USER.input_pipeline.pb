	#J{?/,'@#J{?/,'@!#J{?/,'@	?qۆ?????qۆ????!?qۆ????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#J{?/,'@ A?c?]??AjM??&@Y4??@????*	??????d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????o??!?????uB@)???߾??1??I_?y@@:Preprocessing2F
Iterator::Modely?&1???!9r?YH?@@)bX9?Ȧ?1???EE?:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7?[ A??!?/???C4@)?X?? ??1 Q?#g?1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[????<??!???۔P@)tF??_??1`?:?z?@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;?O???!??C?-?@)Zd;?O???1??C?-?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF%u?{?!?p?%??@)F%u?{?1?p?%??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??r?!??>?*@)/n??r?1??>?*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq?-???!+??r?sC@)F%u?k?1?p?%????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?qۆ????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 A?c?]?? A?c?]??! A?c?]??      ??!       "      ??!       *      ??!       2	jM??&@jM??&@!jM??&@:      ??!       B      ??!       J	4??@????4??@????!4??@????R      ??!       Z	4??@????4??@????!4??@????JCPU_ONLYY?qۆ????b 