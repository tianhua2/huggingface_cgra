# التدريب باستخدام نص برمجى

بالإضافة إلى دفاتر الملاحظات [notebooks](./notebooks) الخاصة بـ 🤗 Transformers، هناك أيضًا نصوص برمجية توضيحية تُظهر كيفية تدريب نموذج لمهمة باستخدام [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch) أو [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow) أو [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax).

كما ستجد النصوص البرمجية التي استخدمناها في [مشاريع الأبحاث](https://github.com/huggingface/transformers/tree/main/examples/research_projects) و [الأمثلة القديمة](https://github.com/huggingface/transformers/tree/main/examples/legacy) والتي ساهم بها المجتمع بشكل أساسي. هذه النصوص البرمجية غير مدعومة بشكل نشط وقد تتطلب إصدارًا محددًا من مكتبة 🤗 Transformers والذي من المحتمل أن يكون غير متوافق مع الإصدار الأحدث من المكتبة.

لا يُتوقع أن تعمل النصوص البرمجية التوضيحية بشكل مباشر على كل مشكلة، وقد تحتاج إلى تكييف النص البرمجي مع المشكلة التي تحاول حلها. ولمساعدتك في ذلك، تعرض معظم النصوص البرمجية كيفية معالجة البيانات قبل التدريب بشكل كامل، مما يتيح لك تحريرها حسب الحاجة لحالتك الاستخدام.

بالنسبة لأي ميزة ترغب في تنفيذها في نص برمجي توضيحي، يرجى مناقشتها في [المنتدى](https://discuss.huggingface.co/) أو في [قضية](https://github.com/huggingface/transformers/issues) قبل إرسال طلب سحب. وفي حين أننا نرحب بإصلاح الأخطاء، فمن غير المرجح أن نقوم بدمج طلب سحب الذي يضيف المزيد من الوظائف على حساب قابلية القراءة.

سيوضح هذا الدليل كيفية تشغيل نص برمجي توضيحي للتدريب على التلخيص في [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) و [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization). يُتوقع أن تعمل جميع الأمثلة مع كلا الإطارين ما لم يُنص على خلاف ذلك.

## الإعداد

لتشغيل الإصدار الأحدث من النصوص البرمجية التوضيحية بنجاح، يجب عليك **تثبيت 🤗 Transformers من المصدر** في بيئة افتراضية جديدة:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

بالنسبة للإصدارات الأقدم من النصوص البرمجية التوضيحية، انقر فوق الزر أدناه:
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

بالنسبة للإصدارات الأقدم من النصوص البرمجية التوضيحية، انقر فوق الزر أدناه:

<details>
  <summary>أمثلة للإصدارات الأقدم من 🤗 Transformers</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

ثم قم بالتبديل إلى النسخة الحالية من 🤗 Transformers إلى إصدار محدد، مثل v3.5.1 على سبيل المثال:

```bash
git checkout tags/v3.5.1
```

بعد إعداد إصدار المكتبة الصحيح، انتقل إلى مجلد الأمثلة الذي تختاره وقم بتثبيت المتطلبات المحددة:

```bash
pip install -r requirements.txt
```

## تشغيل نص برمجي

<frameworkcontent>
<pt>
    
- يقوم النص البرمجي التوضيحي بتنزيل مجموعة بيانات ومعالجتها مسبقًا من مكتبة 🤗 [Datasets](https://huggingface.co/docs/datasets).
- ثم يقوم النص البرمجي بضبط نموذج بيانات دقيق باستخدام [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) على بنية تدعم الملخص. 
- يوضح المثال التالي كيفية ضبط نموذج [T5-small](https://huggingface.co/google-t5/t5-small) على مجموعة بيانات [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail).
- يتطلب نموذج T5 معامل `source_prefix` إضافية بسبب الطريقة التي تم تدريبه بها. يتيح هذا المطالبة لـ T5 معرفة أن هذه مهمة التلخيص.
    
```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
    
- يقوم النص البرمجي التوضيحي بتنزيل مجموعة بيانات ومعالجتها مسبقًا من مكتبة 🤗 [Datasets](https://huggingface.co/docs/datasets/).
- ثم يقوم النص البرمجي بضبط نموذج بيانات دقيق باستخدام Keras على بنية تدعم الملخص.
- يوضح المثال التالي كيفية ضبط نموذج [T5-small](https://huggingface.co/google-t5/t5-small) على مجموعة بيانات [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail).
- يتطلب نموذج T5 ماعمل `source_prefix` إضافية بسبب الطريقة التي تم تدريبه بها. يتيح هذا المطالبة لـ T5 معرفة أن هذه مهمة التلخيص.

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## التدريب الموزع والدقة المختلطة

يدعم [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) التدريب الموزع والدقة المختلطة، مما يعني أنه يمكنك أيضًا استخدامه في نص برمجي. لتمكين كلتا الميزتين:

- أضف معامل `fp16` لتمكين الدقة المختلطة.
- قم بتعيين عدد وحدات معالجة الرسومات (GPUs) التي تريد استخدامها باستخدام حجة `nproc_per_node`.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

تستخدم نصوص TensorFlow البرمجية استراتيجية [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) للتدريب الموزع، ولا تحتاج إلى إضافة أي معامﻻت إضافية إلى النص البرمجي التدريبي. سيستخدم نص TensorFlow البرمجي وحدات معالجة الرسومات (GPUs) متعددة بشكل افتراضي إذا كانت متوفرة.

## تشغيل نص برمجي على وحدة معالجة الدقة الفائقة (TPU)

<frameworkcontent>
<pt>
    
تُعد وحدات معالجة الدقة الفائقة (TPUs) مصممة خصيصًا لتسريع الأداء. يدعم PyTorch وحدات معالجة الدقة الفائقة (TPUs) مع [XLA](https://www.tensorflow.org/xla) مجمع الدقة الفائقة للتعلم العميق (راجع [هنا](https://github.com/pytorch/xla/blob/master/README.md) لمزيد من التفاصيل). لاستخدام وحدة معالجة الدقة الفائقة (TPU)، قم بتشغيل نص `xla_spawn.py` البرمجي واستخدم معامل `num_cores` لتعيين عدد وحدات معالجة الدقة الفائقة (TPU) التي تريد استخدامها.

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
    
تُعد وحدات معالجة الدقة الفائقة (TPUs) مصممة خصيصًا لتسريع الأداء. تستخدم نصوص TensorFlow البرمجية استراتيجية [`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy) للتدريب على وحدات معالجة الدقة الفائقة (TPUs). لاستخدام وحدة معالجة الدقة الفائقة (TPU)، قم بتمرير اسم مورد وحدة معالجة الدقة الفائقة (TPU) إلى حجة `tpu`.
```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## تشغيل نص برمجي باستخدام 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate) هي مكتبة خاصة بـ PyTorch فقط توفر طريقة موحدة لتدريب نموذج على عدة أنواع من الإعدادات (الاعتماد على وحدة المعالجة المركزية (CPU) فقط، أو وحدات معالجة الرسومات (GPUs) المتعددة، أو وحدات معالجة الدقة الفائقة (TPUs)) مع الحفاظ على الرؤية الكاملة لحلقة تدريب PyTorch. تأكد من تثبيت 🤗 Accelerate إذا لم يكن لديك بالفعل:

> ملاحظة: نظرًا لأن Accelerate في حالة تطوير سريع، يجب تثبيت إصدار Git من Accelerate لتشغيل النصوص البرمجية.
```bash
pip install git+https://github.com/huggingface/accelerate
```

بدلاً من نص `run_summarization.py` البرمجي، يجب عليك استخدام نص `run_summarization_no_trainer.py` البرمجي. ستكون النصوص البرمجية المدعومة من 🤗 Accelerate لها ملف `task_no_trainer.py` في المجلد. ابدأ بتشغيل الأمر التالي لإنشاء وحفظ ملف تكوين:

```bash
accelerate config
```

اختبر إعدادك للتأكد من أنه تم تكوينه بشكل صحيح:

```bash
accelerate test
```

الآن أنت مستعد لبدء التدريب:

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn