# مشاركة نموذج

أظهرت آخر درسين تعليميين كيفية ضبط نموذج باستخدام PyTorch و Keras و 🤗 Accelerate لعمليات التهيئة الموزعة. والخطوة التالية هي مشاركة نموذجك مع المجتمع! في Hugging Face، نؤمن بالمشاركة المفتوحة للمعرفة والموارد من أجل إتاحة الذكاء الاصطناعي للجميع. ونشجعك على التفكير في مشاركة نموذجك مع المجتمع لمساعدة الآخرين على توفير الوقت والموارد.

في هذا الدرس، ستتعلم طريقتين لمشاركة نموذج مدرب أو مضبوط على [Model Hub](https://huggingface.co/models):

- قم بالدفع البرمجي لملفاتك إلى Hub.
- قم بسحب وإفلات ملفاتك إلى Hub باستخدام الواجهة web.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="مشغل فيديو YouTube"
frameborder="0" allow="accelerometer; تشغيل تلقائي; لوحة مفاتيح افتراضية; جيروسكوب;
صورة مصغرة" allowfullscreen></iframe>

<Tip>

لمشاركة نموذج مع المجتمع، تحتاج إلى حساب على [huggingface.co](https://huggingface.co/join). يمكنك أيضًا الانضمام إلى منظمة موجودة أو إنشاء منظمة جديدة.

</Tip>

## ميزات المستودع

يعمل كل مستودع على Model Hub مثل مستودع GitHub النموذجي. تقدم مستودعاتنا التحكم في الإصدار وتاريخ الالتزام والقدرة على تصور الاختلافات.

يستند التحكم في إصدار Model Hub المدمج إلى git و [git-lfs](https://git-lfs.github.com/). وبعبارة أخرى، يمكنك التعامل مع نموذج واحد كمستودع واحد، مما يمكّن من زيادة التحكم في الوصول والقابلية للتطوير. يسمح التحكم في الإصدار بتنقيحات، وهي طريقة لتثبيت إصدار محدد من نموذج باستخدام التجزئة أو العلامة أو الفرع.

ونتيجة لذلك، يمكنك تحميل إصدار محدد من النموذج باستخدام معلمة "revision":

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # اسم العلامة، أو اسم الفرع، أو تجزئة الالتزام
... )
```

من السهل أيضًا تحرير الملفات في مستودع، ويمكنك عرض سجل الالتزام بالإضافة إلى الاختلاف:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## الإعداد

قبل مشاركة نموذج على Hub، ستحتاج إلى بيانات اعتماد Hugging Face الخاصة بك. إذا كان لديك حق الوصول إلى المحطة الطرفية، فقم بتشغيل الأمر التالي في بيئة افتراضية حيث تم تثبيت 🤗 Transformers. سيقوم هذا الأمر بتخزين رمز الوصول الخاص بك في مجلد ذاكرة التخزين المؤقت لـ Hugging Face (`~/.cache/` بشكل افتراضي):

```bash
huggingface-cli login
```

إذا كنت تستخدم دفتر ملاحظات مثل Jupyter أو Colaboratory، فتأكد من تثبيت مكتبة [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library). تسمح لك هذه المكتبة بالتفاعل برمجيًا مع Hub.

```bash
pip install huggingface_hub
```

ثم استخدم `notebook_login` لتسجيل الدخول إلى Hub، واتبع الرابط [هنا](https://huggingface.co/settings/token) لإنشاء رمز للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحويل نموذج لجميع الأطر
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحويل نموذج لجميع الأطر

لضمان إمكانية استخدام نموذجك من قبل شخص يعمل بإطار عمل مختلف، نوصي بتحويل وتحميل نموذجك باستخدام نقاط التحقق من PyTorch و TensorFlow. في حين أن المستخدمين لا يزال بإمكانهم تحميل نموذجك من إطار عمل مختلف إذا تخطيت هذه الخطوة، إلا أنه سيكون أبطأ لأن 🤗 Transformers ستحتاج إلى تحويل نقطة التحقق أثناء التنقل.

من السهل تحويل نقطة التحقق لإطار عمل آخر. تأكد من تثبيت PyTorch و TensorFlow (راجع [هنا](installation) لتعليمات التثبيت)، ثم ابحث عن النموذج المحدد لمهمتك في الإطار الآخر.

<frameworkcontent>
<pt>
حدد `from_tf=True` لتحويل نقطة تحقق من TensorFlow إلى PyTorch:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>
حدد `from_pt=True` لتحويل نقطة تحقق من PyTorch إلى TensorFlow:

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

بعد ذلك، يمكنك حفظ نموذج TensorFlow الجديد بنقطة التحقق الجديدة:

```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<jax>
إذا كان النموذج متاحًا في Flax، فيمكنك أيضًا تحويل نقطة تحقق من PyTorch إلى Flax:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## دفع نموذج أثناء التدريب

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

مشاركة نموذج على Hub بسيطة مثل إضافة معلمة أو استدعاء رد اتصال إضافي. تذكر من درس [التدريب الدقيق](training)، أن فئة [`TrainingArguments`] هي المكان الذي تحدد فيه فرط المعلمات وخيارات التدريب الإضافية. تشمل إحدى خيارات التدريب هذه القدرة على دفع نموذج مباشرة إلى Hub. قم بتعيين `push_to_hub=True` في [`TrainingArguments`]:

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

مرر حجج التدريب كالمعتاد إلى [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

بعد ضبط نموذجك، اتصل بـ [`~transformers.Trainer.push_to_hub`] على [`Trainer`] لدفع النموذج المدرب إلى Hub. سوف تضيف 🤗 Transformers تلقائيًا فرط معلمات التدريب ونتائج التدريب وإصدارات الإطار إلى بطاقة النموذج الخاصة بك!

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
شارك نموذجًا على Hub باستخدام [`PushToHubCallback`]. في دالة [`PushToHubCallback`], أضف:

- دليل إخراج لنموذجك.
- رموز.
- `hub_model_id`، والذي هو اسم مستخدم Hub واسم النموذج الخاص بك.

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

أضف الاستدعاء إلى [`fit`](https://keras.io/api/models/model_training_apis/)، وسيقوم 🤗 Transformers بدفع النموذج المدرب إلى Hub:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## استخدام دالة `push_to_hub`

يمكنك أيضًا استدعاء `push_to_hub` مباشرة على نموذجك لتحميله إلى Hub.

حدد اسم نموذجك في `push_to_hub`:

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

ينشئ هذا مستودعًا تحت اسم المستخدم الخاص بك باسم نموذج `my-awesome-model`. يمكن للمستخدمين الآن تحميل نموذجك باستخدام دالة `from_pretrained`:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```
```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

إذا كنت تنتمي إلى منظمة وتريد دفع نموذجك تحت اسم المنظمة بدلاً من ذلك، فما عليك سوى إضافته إلى `repo_id`:

```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

يمكن أيضًا استخدام دالة `push_to_hub` لإضافة ملفات أخرى إلى مستودع النماذج. على سبيل المثال، أضف رموزًا إلى مستودع نموذج:

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

أو ربما تريد إضافة إصدار TensorFlow من نموذج PyTorch المضبوط:

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

الآن عند الانتقال إلى ملفك الشخصي على Hugging Face، يجب أن ترى مستودع النماذج الذي أنشأته حديثًا. سيؤدي النقر فوق علامة التبويب **Files** إلى عرض جميع الملفات التي قمت بتحميلها في المستودع.

للحصول على مزيد من التفاصيل حول كيفية إنشاء الملفات وتحميلها إلى مستودع، راجع وثائق Hub [هنا](https://huggingface.co/docs/hub/how-to-upstream).

## التحميل باستخدام الواجهة web

يمكن للمستخدمين الذين يفضلون نهج عدم الترميز تحميل نموذج من خلال واجهة Hub web. قم بزيارة [huggingface.co/new](https://huggingface.co/new) لإنشاء مستودع جديد:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

من هنا، أضف بعض المعلومات حول نموذجك:

- حدد **مالك** المستودع. يمكن أن يكون هذا أنت أو أي من المنظمات التي تنتمي إليها.
- اختر اسمًا لنموذجك، والذي سيكون أيضًا اسم المستودع.
- اختر ما إذا كان نموذجك عامًا أم خاصًا.
- حدد ترخيص الاستخدام لنموذجك.

الآن انقر فوق علامة التبويب **Files** ثم انقر فوق الزر **Add file** لإضافة ملف جديد إلى مستودعك. ثم اسحب وأسقط ملفًا لتحميله وأضف رسالة الالتزام.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## إضافة بطاقة نموذج

للتأكد من فهم المستخدمين لقدرات نموذجك وقيوده وتحيزاته المحتملة واعتباراته الأخلاقية، يرجى إضافة بطاقة نموذج إلى مستودعك. يتم تعريف بطاقة النموذج في ملف `README.md`. يمكنك إضافة بطاقة نموذج عن طريق:

* قم بإنشاء ملف `README.md` وتحميله يدويًا.
* انقر فوق الزر **Edit model card** في مستودع نموذجك.

الق نظرة على بطاقة [DistilBert](https://huggingface.co/distilbert/distilbert-base-uncased) للحصول على مثال جيد على نوع المعلومات التي يجب أن تتضمنها بطاقة النموذج. للحصول على مزيد من التفاصيل حول الخيارات الأخرى التي يمكنك التحكم فيها في ملف `README.md` مثل البصمة الكربونية للنموذج أو أمثلة الأداة، راجع الوثائق [هنا](https://huggingface.co/docs/hub/models-cards).