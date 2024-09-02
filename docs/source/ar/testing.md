# الاختبار

دعونا نلقي نظرة على كيفية اختبار نماذج 🤗 Transformers وكيف يمكنك كتابة اختبارات جديدة وتحسين الاختبارات الحالية.

هناك مجموعتا اختبار في المستودع:

1. `tests` -- اختبارات لواجهة برمجة التطبيقات العامة
2. `examples` -- اختبارات للتطبيقات المختلفة التي ليست جزءًا من واجهة برمجة التطبيقات

## كيفية اختبار المحولات

1. بمجرد إرسال طلب سحب (PR)، يتم اختباره باستخدام 9 وظائف CircleCi. تتم إعادة اختبار كل التزام جديد لهذا الطلب. يتم تحديد هذه الوظائف في [ملف التكوين](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml)، بحيث يمكنك، إذا لزم الأمر، إعادة إنشاء نفس البيئة على جهازك.

   لا تقوم وظائف CI هذه بتشغيل الاختبارات `@slow`.

2. هناك 3 وظائف يتم تشغيلها بواسطة [GitHub Actions](https://github.com/huggingface/transformers/actions):

   - [تكامل مركز الشعلة](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml): يتحقق مما إذا كان تكامل مركز الشعلة يعمل.
   - [ذاتي الاستضافة (الدفع)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): تشغيل الاختبارات السريعة على وحدة معالجة الرسوميات (GPU) فقط على الالتزامات على `main`. لا يتم تشغيله إلا إذا تم تحديث الالتزام على `main` للرمز في أحد المجلدات التالية: `src`، `tests`، `.github` (لمنع التشغيل عند إضافة بطاقات نموذج، دفاتر الملاحظات، إلخ).
   - [منفذ ذاتي الاستضافة](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): تشغيل الاختبارات العادية والبطيئة على وحدة معالجة الرسوميات (GPU) في `tests` و`examples`:

   ```bash
   RUN_SLOW=1 pytest tests/
   RUN_SLOW=1 pytest examples/
   ```

   يمكن ملاحظة النتائج [هنا](https://github.com/huggingface/transformers/actions).

## تشغيل الاختبارات

### اختيار الاختبارات التي سيتم تشغيلها

تتطرق هذه الوثيقة إلى العديد من التفاصيل حول كيفية تشغيل الاختبارات. إذا كنت بعد قراءة كل شيء، لا تزال بحاجة إلى مزيد من التفاصيل، فستجدها [هنا](https://docs.pytest.org/en/latest/usage.html).

فيما يلي بعض الطرق الأكثر فائدة لتشغيل الاختبارات.

تشغيل الكل:

```console
pytest
```

أو:

```bash
make test
```

لاحظ أن الأخير محدد على النحو التالي:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

الذي يخبر pytest بالقيام بما يلي:

- تشغيل أكبر عدد ممكن من عمليات الاختبار مثل النوى المركزية (التي قد تكون كثيرة جدًا إذا لم يكن لديك الكثير من ذاكرة الوصول العشوائي!)
- التأكد من أن جميع الاختبارات من نفس الملف سيتم تشغيلها بواسطة نفس عملية الاختبار
- لا تلتقط الإخراج
- تشغيل في الوضع التفصيلي

### الحصول على قائمة بجميع الاختبارات

جميع اختبارات مجموعة الاختبار:

```bash
pytest --collect-only -q
```

جميع اختبارات ملف اختبار معين:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### تشغيل وحدة اختبار محددة

لتشغيل وحدة اختبار فردية:

```bash
pytest tests/utils/test_logging.py
```

### تشغيل اختبارات محددة

نظرًا لاستخدام unittest داخل معظم الاختبارات، لتشغيل اختبارات فرعية محددة، تحتاج إلى معرفة اسم فئة unittest التي تحتوي على تلك الاختبارات. على سبيل المثال، يمكن أن يكون:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

هنا:

- `tests/test_optimization.py` - الملف الذي يحتوي على الاختبارات
- `OptimizationTest` - اسم الفئة
- `test_adam_w` - اسم دالة الاختبار المحددة

إذا احتوى الملف على عدة فئات، فيمكنك اختيار تشغيل اختبارات فئة معينة فقط. على سبيل المثال:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

سيقوم بتشغيل جميع الاختبارات داخل تلك الفئة.

كما ذكرنا سابقًا، يمكنك معرفة الاختبارات الموجودة داخل فئة `OptimizationTest` بتشغيل:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

يمكنك تشغيل الاختبارات باستخدام تعبيرات الكلمات الرئيسية.

لتشغيل الاختبارات التي يحتوي اسمها على `adam` فقط:

```bash
pytest -k adam tests/test_optimization.py
```

يمكن استخدام `and` المنطقية و`or` للإشارة إلى ما إذا كان يجب مطابقة جميع الكلمات الرئيسية أو أي منها. يمكن استخدام `not` لنفي.

لتشغيل جميع الاختبارات باستثناء تلك التي يحتوي اسمها على `adam`:

```bash
pytest -k "not adam" tests/test_optimization.py
```

ويمكنك الجمع بين النمطين في واحد:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

على سبيل المثال، لتشغيل كل من `test_adafactor` و`test_adam_w`، يمكنك استخدام:

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

لاحظ أننا نستخدم `or` هنا، لأننا نريد مطابقة أي من الكلمات الرئيسية لإدراج الاثنين.

إذا كنت تريد تضمين الاختبارات التي تحتوي على كلا النمطين فقط، فيجب استخدام `and`:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### تشغيل اختبارات `accelerate`

في بعض الأحيان، تحتاج إلى تشغيل اختبارات `accelerate` على نماذجك. للقيام بذلك، يمكنك فقط إضافة `-m accelerate_tests` إلى أمرك، إذا كنت تريد تشغيل هذه الاختبارات على `OPT`، فقم بتشغيل:

```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py
```

### تشغيل اختبارات التوثيق

لاختبار ما إذا كانت أمثلة التوثيق صحيحة، يجب التأكد من أن `doctests` ناجحة.

كمثال، دعنا نستخدم [docstring لـ `WhisperModel.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035):

```python
r"""
Returns:

Example:
```python
>>> import torch
>>> from transformers import WhisperModel, WhisperFeatureExtractor
>>> from datasets import load_dataset
>>> model = WhisperModel.from_pretrained("openai/whisper-base")
>>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 512]
```"""
```

قم ببساطة بتشغيل السطر التالي لاختبار كل مثال docstring تلقائيًا في الملف المطلوب:

```bash
pytest --doctest-modules <path_to_file_or_dir>
```

إذا كان للملف ملحق markdown، فيجب إضافة الحجة `--doctest-glob="*.md"`.

### تشغيل الاختبارات المعدلة فقط

يمكنك تشغيل الاختبارات المتعلقة بالملفات أو الفروع غير المرحلية (وفقًا لـ Git) باستخدام [pytest-picked](https://github.com/anapaulagomes/pytest-picked). هذه طريقة رائعة لاختبار ما إذا كانت التغييرات التي أجريتها لم تكسر أي شيء بسرعة، حيث لن تقوم بتشغيل الاختبارات المتعلقة بالملفات التي لم تلمسها.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

سيتم تشغيل جميع الاختبارات من الملفات والمجلدات التي تم تعديلها ولكن لم يتم ارتكابها بعد.
: هذا دليل سريع حول كيفية تشغيل اختباراتنا.

### إعادة تشغيل الاختبارات التي فشلت تلقائيًا عند تعديل المصدر
يوفر [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) ميزة مفيدة جدًا تتمثل في اكتشاف جميع الاختبارات التي فشلت، ثم انتظار تعديل الملفات وإعادة تشغيل الاختبارات التي فشلت باستمرار حتى تنجح أثناء إصلاحها. بحيث لا تحتاج إلى إعادة تشغيل pytest بعد إجراء الإصلاح. ويستمر هذا حتى نجاح جميع الاختبارات، وبعد ذلك يتم إجراء تشغيل كامل مرة أخرى.

```bash
pip install pytest-xdist
```

للدخول إلى الوضع: `pytest -f` أو `pytest --looponfail`

يتم اكتشاف تغييرات الملف عن طريق النظر في دلائل الجذر looponfailroots ومحتوياتها بالكامل (بشكل متكرر). إذا لم ينجح الافتراضي لهذه القيمة، فيمكنك تغييره في مشروعك عن طريق تعيين خيار تكوين في setup.cfg:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

أو ملفات pytest.ini/tox.ini:

```ini
[pytest]
looponfailroots = transformers tests
```

سيؤدي هذا إلى البحث فقط عن تغييرات الملفات في دلائل المقابلة، المحددة نسبيًا إلى دليل ملف ini.

[pytest-watch](https://github.com/joeyespo/pytest-watch) هو تنفيذ بديل لهذه الوظيفة.

### تخطي نموذج الاختبار
إذا كنت تريد تشغيل جميع نماذج الاختبار، باستثناء بعضها، فيمكنك استبعادها عن طريق إعطاء قائمة صريحة بالاختبارات التي سيتم تشغيلها. على سبيل المثال، لتشغيل الكل باستثناء اختبارات test_modeling_*.py:

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### مسح الحالة
يجب مسح ذاكرة التخزين المؤقت في عمليات البناء CI وعندما تكون العزلة مهمة (ضد السرعة):

```bash
pytest --cache-clear tests
```

### تشغيل الاختبارات بالتوازي
كما ذكر سابقًا، يقوم make test بتشغيل الاختبارات بالتوازي عبر المكون الإضافي pytest-xdist (`-n X` argument، على سبيل المثال `-n 2` لتشغيل وظيفتين متوازيتين).

يسمح خيار `--dist=` في pytest-xdist بالتحكم في كيفية تجميع الاختبارات. ويضع `--dist=loadfile` الاختبارات الموجودة في ملف واحد في نفس العملية.

نظرًا لأن ترتيب الاختبارات المنفذة مختلف ولا يمكن التنبؤ به، إذا أدى تشغيل مجموعة الاختبارات باستخدام pytest-xdist إلى حدوث فشل (مما يعني وجود بعض الاختبارات المقترنة غير المكتشفة)، فاستخدم [pytest-replay](https://github.com/ESSS/pytest-replay) لإعادة تشغيل الاختبارات بنفس الترتيب، والذي يجب أن يساعد في تقليل تسلسل الفشل هذا إلى الحد الأدنى.

### ترتيب الاختبار والتكرار
من الجيد تكرار الاختبارات عدة مرات، بالتسلسل، أو بشكل عشوائي، أو في مجموعات، للكشف عن أي أخطاء محتملة في التبعية المتبادلة والمتعلقة بالحالة (الإنهاء). والتكرار المتعدد المباشر جيد للكشف عن بعض المشكلات التي تكشفها الطبيعة العشوائية لتعلم الآلة.

#### تكرار الاختبارات
- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

ثم قم بتشغيل كل اختبار عدة مرات (50 بشكل افتراضي):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>
هذه الإضافة لا تعمل مع العلامة -n من pytest-xdist.
</Tip>

<Tip>
هناك إضافة أخرى تسمى pytest-repeat، لكنها لا تعمل مع unittest.
</Tip>

#### تشغيل الاختبارات بترتيب عشوائي
```bash
pip install pytest-random-order
```

مهم: سيؤدي وجود pytest-random-order إلى جعل الاختبارات عشوائية تلقائيًا، دون الحاجة إلى إجراء أي تغييرات في التكوين أو خيارات سطر الأوامر.

كما هو موضح سابقًا، يسمح هذا باكتشاف الاختبارات المقترنة - حيث تؤثر حالة أحد الاختبارات على حالة اختبار آخر. عندما يتم تثبيت pytest-random-order، فإنه سيطبع البذرة العشوائية التي استخدمها لتلك الجلسة، على سبيل المثال:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

حتى أنه إذا فشلت تسلسل معين، فيمكنك إعادة إنتاجه عن طريق إضافة تلك البذرة الدقيقة، على سبيل المثال:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

سيؤدي ذلك إلى إعادة إنتاج الترتيب الدقيق فقط إذا كنت تستخدم نفس قائمة الاختبارات (أو عدم وجود قائمة على الإطلاق). وبمجرد أن تبدأ في تضييق قائمة الاختبارات يدويًا، فلن تتمكن بعد ذلك من الاعتماد على البذرة، ولكن يجب عليك إدراجها يدويًا بالترتيب الدقيق الذي فشلت فيه وإخبار pytest بعدم جعلها عشوائية بدلاً من ذلك باستخدام `--random-order-bucket=none`، على سبيل المثال:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

لإيقاف التشويش لجميع الاختبارات:

```bash
pytest --random-order-bucket=none
```

بشكل افتراضي، يتم ضمنيًا استخدام `--random-order-bucket=module`، والذي سيقوم بخلط الملفات على مستوى الوحدات النمطية. ويمكنه أيضًا الخلط على مستويات "class" و"package" و"global" و"none". للحصول على التفاصيل الكاملة، يرجى الاطلاع على [وثائقه](https://github.com/jbasko/pytest-random-order).

بديل آخر للعشوائية هو: [`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly). لهذه الوحدة النمطية وظائف/واجهة مشابهة جدًا، ولكنها لا تحتوي على أوضاع الدلاء المتاحة في pytest-random-order. ولديها نفس المشكلة في فرض نفسها بمجرد تثبيتها.

### الاختلافات في المظهر والشعور
#### pytest-sugar
[pytest-sugar](https://github.com/Frozenball/pytest-sugar) هي إضافة تحسن المظهر والشعور، وتضيف شريط تقدم، وتظهر الاختبارات التي تفشل والتأكيد على الفور. يتم تنشيطه تلقائيًا عند التثبيت.

```bash
pip install pytest-sugar
```

لتشغيل الاختبارات بدونها، قم بتشغيل:

```bash
pytest -p no:sugar
```

أو إلغاء تثبيتها.

#### الإبلاغ عن اسم كل اختبار فرعي وتقدمه
لاختبار واحد أو مجموعة من الاختبارات عبر pytest (بعد pip install pytest-pspec):

```bash
pytest --pspec tests/test_optimization.py
```

#### إظهار الاختبارات الفاشلة على الفور
[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) تعرض حالات الفشل والأخطاء على الفور بدلاً من الانتظار حتى نهاية جلسة الاختبار.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```
## إلى GPU أو عدم التوجه إلى GPU

على إعداد ممكّن لـ GPU، لاختبار الوضع الخاص بـ CPU فقط، أضف "CUDA_VISIBLE_DEVICES=":

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

أو إذا كان لديك عدة وحدات GPU، فيمكنك تحديد أيها سيتم استخدامه بواسطة "pytest". على سبيل المثال، لاستخدام وحدة GPU الثانية فقط إذا كان لديك وحدات GPU "0" و"1"، يمكنك تشغيل ما يلي:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

هذا مفيد عندما تريد تشغيل مهام مختلفة على وحدات GPU مختلفة.

يجب تشغيل بعض الاختبارات على وضع CPU فقط، بينما يمكن تشغيل اختبارات أخرى على CPU أو GPU أو TPU، وهناك اختبارات أخرى يجب تشغيلها على وحدات GPU متعددة. ويتم استخدام زخارف التخطي التالية لتحديد متطلبات الاختبارات فيما يتعلق بـ CPU/GPU/TPU:

- `require_torch` - سيتم تشغيل هذا الاختبار فقط تحت torch

- `require_torch_gpu` - مثل `require_torch` بالإضافة إلى أنه يتطلب وحدة GPU واحدة على الأقل

- `require_torch_multi_gpu` - مثل `require_torch` بالإضافة إلى أنه يتطلب وحدتي GPU على الأقل

- `require_torch_non_multi_gpu` - مثل `require_torch` بالإضافة إلى أنه يتطلب صفر أو وحدة GPU واحدة

- `require_torch_up_to_2_gpus` - مثل `require_torch` بالإضافة إلى أنه يتطلب صفر أو وحدة أو وحدتي GPU

- `require_torch_xla` - مثل `require_torch` بالإضافة إلى أنه يتطلب وحدة TPU واحدة على الأقل

دعونا نوضح متطلبات وحدة GPU في الجدول التالي:

| عدد وحدات GPU | الزخرف                         |
|---------|----------------------------|
| >= 0   | `@require_torch`           |
| >= 1   | `@require_torch_gpu`       |
| >= 2   | `@require_torch_multi_gpu` |
| < 2    | `@require_torch_non_multi_gpu` |
| < 3    | `@require_torch_up_to_2_gpus`  |

على سبيل المثال، إليك اختبار يجب تشغيله فقط عندما تكون هناك وحدتي GPU أو أكثر متاحتين وتم تثبيت pytorch:

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

إذا تطلب الاختبار وجود `tensorflow`، فيجب استخدام الزخرف `require_tf`. على سبيل المثال:

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

يمكن تكديس هذه الزخارف فوق بعضها البعض. على سبيل المثال، إذا كان الاختبار بطيئًا ويتطلب وحدة GPU واحدة على الأقل في pytorch، فيمكنك إعداده على النحو التالي:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

يقوم بعض الزخارف مثل `@parametrized` بإعادة كتابة أسماء الاختبارات، لذلك يجب إدراج زخارف التخطي `@require_*` في النهاية لكي تعمل بشكل صحيح. وفيما يلي مثال على الاستخدام الصحيح:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

لا توجد هذه المشكلة في الترتيب مع `@pytest.mark.parametrize`، فيمكنك وضعه أولاً أو آخرًا وسيظل يعمل. ولكنه يعمل فقط مع non-unittests.

داخل الاختبارات:

- عدد وحدات GPU المتاحة:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count() # يعمل مع torch و tf
```

## الاختبار باستخدام PyTorch خلفية أو جهاز محدد

لتشغيل مجموعة الاختبارات على جهاز PyTorch محدد، أضف `TRANSFORMERS_TEST_DEVICE="$device"` حيث `$device` هو backend المستهدف. على سبيل المثال، لاختبار وضع CPU فقط:

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

تكون هذه المتغيرات مفيدة لاختبار backends PyTorch مخصصة أو أقل شيوعًا مثل `mps` أو `xpu` أو `npu`. ويمكن أيضًا استخدامها لتحقيق نفس تأثير `CUDA_VISIBLE_DEVICES` عن طريق استهداف وحدات GPU محددة أو الاختبار في وضع CPU فقط.

قد تتطلب بعض الأجهزة استيرادًا إضافيًا بعد استيراد `torch` للمرة الأولى. ويمكن تحديد هذا باستخدام متغير البيئة `TRANSFORMERS_TEST_BACKEND`:

```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```

قد تتطلب backends البديلة أيضًا استبدال وظائف محددة للجهاز. على سبيل المثال، قد تحتاج إلى استبدال `torch.cuda.manual_seed` بوظيفة تعيين البذور المحددة للجهاز مثل `torch.npu.manual_seed` أو `torch.xpu.manual_seed` لتعيين البذور العشوائية بشكل صحيح على الجهاز. ولتحديد backend جديد مع وظائف محددة للجهاز عند تشغيل مجموعة الاختبارات، قم بإنشاء ملف مواصفات Python باسم `spec.py` بالتنسيق التالي:

```python
import torch
import torch_npu # بالنسبة إلى xpu، استبدله بـ `import intel_extension_for_pytorch`
# !! يمكن إضافة استيرادات إضافية هنا !!

# تحديد اسم الجهاز (مثل 'cuda' أو 'cpu' أو 'npu' أو 'xpu' أو 'mps')
DEVICE_NAME = 'npu'

# تحديد backends المحددة للجهاز لإرسالها.
# إذا لم يتم تحديدها، فسيتم الرجوع إلى 'default' في 'testing_utils.py`
MANUAL_SEED_FN = torch.npu.manual_seed
EMPTY_CACHE_FN = torch.npu.empty_cache
DEVICE_COUNT_FN = torch.npu.device_count
```

يسمح هذا التنسيق أيضًا بتحديد أي استيرادات إضافية مطلوبة. لاستخدام هذا الملف لاستبدال الطرق المكافئة في مجموعة الاختبارات، قم بتعيين متغير البيئة `TRANSFORMERS_TEST_DEVICE_SPEC` إلى مسار ملف المواصفات، على سبيل المثال `TRANSFORMERS_TEST_DEVICE_SPEC=spec.py`.

حاليًا، يتم دعم `MANUAL_SEED_FN` و`EMPTY_CACHE_FN` و`DEVICE_COUNT_FN` فقط للتفويض المحدد للجهاز.

## التدريب الموزع

لا يمكن لـ `pytest` التعامل مع التدريب الموزع مباشرةً. إذا تم محاولة ذلك - فإن العمليات الفرعية لا تقوم بالشيء الصحيح وتنتهي بالتفكير في أنها `pytest` وتبدأ في تشغيل مجموعة الاختبارات في حلقات. ولكنه يعمل إذا قام أحد بتشغيل عملية عادية تقوم بعد ذلك بتشغيل العديد من العمال وإدارة أنابيب الإدخال/الإخراج.

وفيما يلي بعض الاختبارات التي تستخدمها:

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)

- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

للانتقال مباشرةً إلى نقطة التنفيذ، ابحث عن مكالمة `execute_subprocess_async` في تلك الاختبارات.

ستحتاج إلى وحدتي GPU على الأقل لمشاهدة هذه الاختبارات أثناء العمل:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

## التقاط الإخراج

أثناء تنفيذ الاختبار، يتم التقاط أي إخراج يتم إرساله إلى `stdout` و`stderr`. إذا فشل أحد الاختبارات أو طريقة الإعداد، فسيتم عادةً عرض الإخراج المقابل الذي تم التقاطه جنبًا إلى جنب مع تتبع فشل.

لإيقاف التقاط الإخراج والحصول على `stdout` و`stderr` بشكل طبيعي، استخدم `-s` أو `--capture=no`:

```bash
pytest -s tests/utils/test_logging.py
```

لإرسال نتائج الاختبار إلى إخراج بتنسيق JUnit:

```bash
pytest tests --junitxml=result.xml
```

## التحكم في الألوان

لعدم استخدام أي لون (على سبيل المثال، الأصفر على خلفية بيضاء غير قابل للقراءة):

```bash
pytest --color=no tests/utils/test_logging.py
```

## إرسال تقرير الاختبار إلى خدمة Pastebin عبر الإنترنت

إنشاء عنوان URL لكل فشل في الاختبار:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

سيقوم هذا بإرسال معلومات تشغيل الاختبار إلى خدمة Paste عن بُعد وتوفير عنوان URL لكل فشل. يمكنك اختيار الاختبارات كالمعتاد أو إضافة -x إذا كنت تريد إرسال فشل واحد فقط على سبيل المثال.

إنشاء عنوان URL لسجل جلسة الاختبار بالكامل:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## كتابة الاختبارات

تعتمد اختبارات 🤗 transformers على `unittest`، ولكن يتم تشغيلها بواسطة `pytest`، لذلك يمكن استخدام ميزات كلا النظامين في معظم الوقت.

يمكنك قراءة [هنا](https://docs.pytest.org/en/stable/unittest.html) عن الميزات المدعومة، ولكن الشيء المهم الذي يجب تذكره هو أن معظم مؤشرات `pytest` لا تعمل. ولا يتم دعم المعلمات أيضًا، ولكننا نستخدم الوحدة النمطية `parameterized` التي تعمل بطريقة مشابهة.
### المعلمة 

غالبًا ما تكون هناك حاجة لتشغيل نفس الاختبار عدة مرات، ولكن باستخدام حجج مختلفة. يمكن القيام بذلك من داخل الاختبار، ولكن بعد ذلك لا توجد طريقة لتشغيل هذا الاختبار لمجموعة واحدة فقط من الحجج. 

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
    [
    ("negative", -1.5, -2.0),
    ("integer", 1, 1.0),
    ("large fraction", 1.6, 1),
    ]
    )
    def test_floor(self, name, input, expected):
    assert_equal(math.floor(input), expected)
``` 

الآن، بشكل افتراضي، سيتم تشغيل هذا الاختبار 3 مرات، وفي كل مرة يتم تعيين الحجج الثلاث الأخيرة لـ `test_floor` إلى الحجج المقابلة في قائمة المعلمات. 

ويمكنك تشغيل مجموعات "negative" و "integer" فقط من المعلمات باستخدام: 

```bash
pytest -k "negative and integer" tests/test_mytest.py
``` 

أو جميع المجموعات الفرعية باستثناء "negative" باستخدام: 

```bash
pytest -k "not negative" tests/test_mytest.py
``` 

بالإضافة إلى استخدام عامل تصفية `-k` المذكور أعلاه، يمكنك معرفة الاسم الدقيق لكل اختبار فرعي وتشغيل أي منها أو جميعها باستخدام أسمائها الدقيقة. 

```bash
pytest test_this1.py --collect-only -q
``` 

وسوف يسرد: 

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
``` 

لذا الآن يمكنك تشغيل اختبارين فرعيين محددين فقط: 

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative test_this1.py::TestMathUnitTest::test_floor_1_integer
``` 

تتوافق الوحدة النمطية [parameterized](https://pypi.org/project/parameterized/)، الموجودة بالفعل في تبعيات المطور في `transformers`، مع كلا النوعين: `unittests` و `pytest` tests. 

ومع ذلك، إذا لم يكن الاختبار عبارة عن `unittest`، فيمكنك استخدام `pytest.mark.parametrize` (أو قد ترى أنها مستخدمة في بعض الاختبارات الموجودة، خاصةً تحت `examples`). 

هنا نفس المثال، ولكن هذه المرة باستخدام مؤشر `pytest` `parametrize`: 

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
"name, input, expected",
[
("negative", -1.5, -2.0),
("integer", 1, 1.0),
("large fraction", 1.6, 1),
],
)
def test_floor(name, input, expected):
assert_equal(math.floor(input), expected)
``` 

مثل `parameterized`، يمكنك التحكم الدقيق في الاختبارات الفرعية التي يتم تشغيلها باستخدام `pytest.mark.parametrize` إذا لم تقم وظيفة التصفية `-k` بالمهمة. باستثناء أن دالة المعلمة هذه تُنشئ مجموعة مختلفة قليلاً من الأسماء للاختبارات الفرعية. فيما يلي كيفية ظهورها: 

```bash
pytest test_this2.py --collect-only -q
``` 

وسوف يسرد: 

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
``` 

لذا الآن يمكنك تشغيل الاختبار المحدد فقط: 

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
``` 

كما في المثال السابق. 

### الملفات والمجلدات 

في الاختبارات، نحتاج غالبًا إلى معرفة مكان وجود الأشياء بالنسبة لملف الاختبار الحالي، وهذا ليس أمرًا بسيطًا لأن الاختبار قد يتم استدعاؤه من أكثر من دليل أو قد يكون موجودًا في مجلدات فرعية بدرجات مختلفة. تقوم فئة المساعدة `transformers.test_utils.TestCasePlus` بحل هذه المشكلة عن طريق فرز جميع المسارات الأساسية وتوفير وصول سهل إليها: 

- كائنات `pathlib` (جميعها محددة بالكامل): 

- `test_file_path` - مسار ملف الاختبار الحالي، أي `__file__`

- `test_file_dir` - الدليل الذي يحتوي على ملف الاختبار الحالي 

- `tests_dir` - دليل مجموعة اختبارات `tests` 

- `examples_dir` - دليل مجموعة اختبارات `examples` 

- `repo_root_dir` - دليل مستودع 

- `src_dir` - دليل `src` (أي حيث يوجد المجلد الفرعي `transformers`) 

- المسارات المعبرة كسلاسل - نفس ما سبق ولكن هذه الأساليب تعيد المسارات كسلاسل، بدلاً من كائنات `pathlib`: 

- `test_file_path_str` 

- `test_file_dir_str` 

- `tests_dir_str` 

- `examples_dir_str` 

- `repo_root_dir_str` 

- `src_dir_str` 

لبدء استخدام هذه الطرق، كل ما تحتاجه هو التأكد من أن الاختبار موجود في فئة فرعية من `transformers.test_utils.TestCasePlus`. على سبيل المثال: 

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
def test_something_involving_local_locations(self):
data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
``` 

إذا لم تكن بحاجة إلى التعامل مع المسارات عبر `pathlib` أو إذا كنت بحاجة فقط إلى مسار كسلسلة، فيمكنك دائمًا استدعاء `str()` على كائن `pathlib` أو استخدام الأساليب التي تنتهي بـ `_str`. على سبيل المثال: 

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
def test_something_involving_stringified_locations(self):
examples_dir = self.examples_dir_str
``` 

### الملفات والمجلدات المؤقتة 

يعد استخدام الملفات والمجلدات المؤقتة الفريدة أمرًا ضروريًا لتشغيل الاختبارات بشكل متوازي، بحيث لا تكتب الاختبارات فوق بيانات بعضها البعض. كما نريد حذف الملفات والمجلدات المؤقتة في نهاية كل اختبار قام بإنشائها. لذلك، من الضروري استخدام حزم مثل `tempfile`، والتي تلبي هذه الاحتياجات. 

ومع ذلك، عند تصحيح أخطاء الاختبارات، تحتاج إلى القدرة على رؤية ما يتم إدخاله في الملف أو الدليل المؤقت وتريد معرفة مساره الدقيق وعدم جعله عشوائيًا في كل مرة يتم فيها إعادة تشغيل الاختبار. 

تعد فئة المساعدة `transformers.test_utils.TestCasePlus` أفضل للاستخدام في مثل هذه الأغراض. إنها فئة فرعية من `unittest.TestCase`، لذلك يمكننا بسهولة أن نرث منها في وحدات الاختبار. 

هنا مثال على استخدامها: 

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
def test_whatever(self):
tmp_dir = self.get_auto_remove_tmp_dir()
``` 

ينشئ هذا الكود دليلًا مؤقتًا فريدًا، ويحدد `tmp_dir` إلى موقعه. 

- إنشاء دليل مؤقت فريد: 

```python
def test_whatever(self):
tmp_dir = self.get_auto_remove_tmp_dir()
``` 

سيحتوي `tmp_dir` على المسار إلى الدليل المؤقت الذي تم إنشاؤه. سيتم إزالته تلقائيًا في نهاية الاختبار. 

- إنشاء دليل مؤقت من اختياري، والتأكد من أنه فارغ قبل بدء الاختبار وعدم إفراغه بعد الاختبار. 

```python
def test_whatever(self):
tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
``` 

هذا مفيد للتصحيح عندما تريد مراقبة دليل محدد وتريد التأكد من أن الاختبارات السابقة لم تترك أي بيانات هناك. 

- يمكنك تجاوز السلوك الافتراضي عن طريق تجاوز وسيطي `before` و `after` مباشرةً، مما يؤدي إلى أحد السلوكيات التالية: 

- `before=True`: سيتم دائمًا مسح الدليل المؤقت في بداية الاختبار. 

- `before=False`: إذا كان الدليل المؤقت موجودًا بالفعل، فستبقى أي ملفات موجودة فيه. 

- `after=True`: سيتم دائمًا حذف الدليل المؤقت في نهاية الاختبار. 

- `after=False`: سيتم دائمًا ترك الدليل المؤقت دون تغيير في نهاية الاختبار. 

<Tip> 

لتشغيل ما يعادل `rm -r` بأمان، لا يُسمح إلا بالدلائل الفرعية لمستخرج مستودع المشروع إذا تم استخدام `tmp_dir` صريح، بحيث لا يتم عن طريق الخطأ مسح أي جزء مهم من نظام الملفات مثل `/tmp` أو ما شابه ذلك. يرجى دائمًا تمرير المسارات التي تبدأ بـ `./`. 

</Tip> 

<Tip> 

يمكن لكل اختبار تسجيل عدة مجلدات مؤقتة وسيتم إزالتها جميعًا تلقائيًا، ما لم يُطلب خلاف ذلك. 

</Tip> 

### التغلب المؤقت على sys.path 

إذا كنت بحاجة إلى تجاوز `sys.path` مؤقتًا لاستيراد اختبار آخر، على سبيل المثال، فيمكنك استخدام مدير السياق `ExtendSysPath`. مثال: 

```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
from test_trainer import TrainerIntegrationCommon  # noqa
```
### تجاوز الاختبارات

هذا مفيد عندما يتم اكتشاف خطأ ويتم كتابة اختبار جديد، ولكن لم يتم إصلاح الخطأ بعد. حتى نتمكن من الالتزام به في المستودع الرئيسي، يجب أن نتأكد من أنه تم تخطيه أثناء `make test`.

الطرق:

-  **تجاوز** يعني أنك تتوقع أن يمر اختبارك فقط إذا تم استيفاء بعض الشروط، وإلا يجب على pytest أن يتجاوز تشغيل الاختبار بالكامل. ومن الأمثلة الشائعة على ذلك تجاوز الاختبارات الخاصة بنظام Windows فقط على منصات غير Windows، أو تجاوز الاختبارات التي تعتمد على مورد خارجي غير متوفر في الوقت الحالي (مثل قاعدة بيانات).

-  **xfail** يعني أنك تتوقع فشل الاختبار لسبب ما. ومن الأمثلة الشائعة على ذلك اختبار ميزة لم يتم تنفيذها بعد، أو خطأ لم يتم إصلاحه بعد. عندما ينجح الاختبار على الرغم من توقع فشله (تمت تسميته بـ pytest.mark.xfail)، فهو xpass وسيتم الإبلاغ عنه في ملخص الاختبار.

أحد الاختلافات المهمة بين الاثنين هو أن `skip` لا يشغل الاختبار، و`xfail` يفعل ذلك. لذا إذا كان الكود المعيب يتسبب في حالة سيئة ستؤثر على الاختبارات الأخرى، فلا تستخدم `xfail`.

#### التنفيذ

-  إليك كيفية تخطي اختبار كامل دون شروط:

```python no-style
@unittest.skip(reason="this bug needs to be fixed")
def test_feature_x():
```

أو عبر pytest:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

أو بطريقة `xfail`:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

فيما يلي كيفية تخطي اختبار بناءً على فحوصات داخلية داخل الاختبار:

```python
def test_feature_x():
if not has_something():
pytest.skip("unsupported configuration")
```

أو الوحدة النمطية بأكملها:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

أو بطريقة `xfail`:

```python
def test_feature_x():
pytest.xfail("expected to fail until bug XYZ is fixed")
```

-  فيما يلي كيفية تخطي جميع الاختبارات في وحدة نمطية إذا كان هناك استيراد مفقود:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

-  تخطي اختبار بناءً على شرط:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

أو:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

أو تخطي الوحدة النمطية بأكملها:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
def test_feature_x(self):
```

لمزيد من التفاصيل والأمثلة والطرق، راجع [هنا](https://docs.pytest.org/en/latest/skipping.html).

### الاختبارات البطيئة

مكتبة الاختبارات تتزايد باستمرار، ويستغرق بعض الاختبارات دقائق للتشغيل، لذلك لا يمكننا تحمل الانتظار لمدة ساعة حتى تكتمل مجموعة الاختبارات على CI. لذلك، مع بعض الاستثناءات للاختبارات الأساسية، يجب وضع علامة على الاختبارات البطيئة كما هو موضح في المثال أدناه:

```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

بمجرد وضع علامة على الاختبار على أنه `@slow`، لتشغيل هذه الاختبارات، قم بتعيين متغير البيئة `RUN_SLOW=1`، على سبيل المثال:

```bash
RUN_SLOW=1 pytest tests
```

يقوم بعض الديكورات مثل `@parameterized` بإعادة كتابة أسماء الاختبارات، لذلك يجب إدراج `@slow` وبقية الديكورات `@require_*` في النهاية حتى تعمل بشكل صحيح. فيما يلي مثال على الاستخدام الصحيح:

```python no-style
@parameterized.expand(...)
@slow
def test_integration_foo():
```

كما هو موضح في بداية هذه الوثيقة، يتم تشغيل الاختبارات البطيئة وفقًا لجدول زمني، بدلاً من فحوصات CI في PRs. لذلك من الممكن أن يتم تفويت بعض المشكلات أثناء تقديم طلب سحب وسيتم دمجها. سيتم اكتشاف هذه المشكلات أثناء مهمة CI المجدولة التالية. ولكن هذا يعني أيضًا أنه من المهم تشغيل الاختبارات البطيئة على جهازك قبل تقديم طلب السحب.

فيما يلي آلية صنع القرار لاختيار الاختبارات التي يجب وضع علامة عليها على أنها بطيئة:

إذا كان الاختبار يركز على أحد المكونات الداخلية للمكتبة (على سبيل المثال، ملفات النمذجة، أو ملفات التمييز، أو الأنابيب)، فيجب علينا تشغيل هذا الاختبار في مجموعة الاختبارات غير البطيئة. إذا كان يركز على جانب آخر من جوانب المكتبة، مثل الوثائق أو الأمثلة، فيجب علينا تشغيل هذه الاختبارات في مجموعة الاختبارات البطيئة. وبعد ذلك، لتنقيح هذا النهج، يجب أن تكون لدينا استثناءات:

-  يجب وضع علامة على جميع الاختبارات التي تحتاج إلى تنزيل مجموعة كبيرة من الأوزان أو مجموعة بيانات أكبر من ~50 ميجابايت (على سبيل المثال، اختبارات تكامل النماذج أو المحلل اللغوي أو الأنابيب) على أنها بطيئة. إذا كنت تقوم بإضافة نموذج جديد، فيجب عليك إنشاء وتحميل إصدار مصغر منه (بأوزان عشوائية) لاختبارات التكامل. يتم مناقشة ذلك في الفقرات التالية.

-  يجب وضع علامة على جميع الاختبارات التي تحتاج إلى إجراء تدريب غير مُستَهدف بشكل خاص ليكون سريعًا على أنها بطيئة.

-  يمكننا تقديم استثناءات إذا كانت بعض هذه الاختبارات التي يجب ألا تكون بطيئة بطيئة للغاية، ووضع علامة عليها على أنها `@slow`. تعد اختبارات النمذجة التلقائية، التي تقوم بحفظ وتحميل ملفات كبيرة على القرص، مثالًا جيدًا على الاختبارات التي تم وضع علامة عليها على أنها `@slow`.

-  إذا اكتمل الاختبار في أقل من ثانية واحدة على CI (بما في ذلك عمليات التنزيل إن وجدت)، فيجب أن يكون اختبارًا طبيعيًا بغض النظر عن ذلك.

بشكل جماعي، يجب أن تغطي جميع الاختبارات غير البطيئة المكونات الداخلية المختلفة بالكامل، مع الحفاظ على سرعتها. على سبيل المثال، يمكن تحقيق تغطية كبيرة من خلال الاختبار باستخدام نماذج مصغرة تم إنشاؤها خصيصًا بأوزان عشوائية. تحتوي هذه النماذج على الحد الأدنى من عدد الطبقات (على سبيل المثال، 2)، وحجم المفردات (على سبيل المثال، 1000)، وما إلى ذلك. بعد ذلك، يمكن لاختبارات `@slow` استخدام نماذج كبيرة وبطيئة لإجراء اختبارات نوعية. لمشاهدة استخدام هذه، ما عليك سوى البحث عن النماذج *tiny* باستخدام:

```bash
grep tiny tests examples
```

فيما يلي مثال على [script](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py) الذي أنشأ النموذج المصغر [stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de). يمكنك ضبطه بسهولة على الهندسة المعمارية المحددة لنموذجك.

من السهل قياس وقت التشغيل بشكل غير صحيح إذا كان هناك، على سبيل المثال، إشراف على تنزيل نموذج ضخم، ولكن إذا قمت باختباره محليًا، فسيتم تخزين الملفات التي تم تنزيلها مؤقتًا وبالتالي لن يتم قياس وقت التنزيل. لذلك، تحقق من تقرير سرعة التنفيذ في سجلات CI بدلاً من ذلك (إخراج `pytest --durations=0 tests`).

هذا التقرير مفيد أيضًا للعثور على القيم الشاذة البطيئة التي لم يتم وضع علامة عليها على هذا النحو، أو التي تحتاج إلى إعادة كتابتها لتكون سريعة. إذا لاحظت أن مجموعة الاختبارات بدأت تصبح بطيئة على CI، فسيظهر أعلى قائمة بهذا التقرير أبطأ الاختبارات.
## اختبار إخراج stdout/stderr

لاختبار الدوال التي تكتب في stdout و/أو stderr، يمكن للاختبار الوصول إلى هذه التدفقات باستخدام نظام capsys في pytest كما يلي:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # استهلاك تدفقات الإخراج التي تم التقاطها
    # اختياري: إذا كنت تريد إعادة تشغيل التدفقات التي تم استهلاكها:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # الاختبار:
    assert msg in out
    assert msg in err
```

وبالطبع، في معظم الأحيان، ستأتي stderr كجزء من استثناء، لذلك يجب استخدام try/except في هذه الحالة:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
    assert msg in error, f"{msg} موجود في الاستثناء:\n{error}"
```

هناك طريقة أخرى لالتقاط stdout وهي عبر contextlib.redirect_stdout:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # اختياري: إذا كنت تريد إعادة تشغيل التدفقات التي تم استهلاكها:
    sys.stdout.write(out)
    # الاختبار:
    assert msg in out
```

هناك مشكلة محتملة مهمة عند التقاط stdout وهي أنها قد تحتوي على أحرف \r التي تقوم في عملية الطباعة العادية بإعادة تعيين كل ما تم طباعته حتى الآن. لا توجد مشكلة مع pytest، ولكن مع pytest -s، يتم تضمين هذه الأحرف في المخزن المؤقت، لذلك لكي تتمكن من تشغيل الاختبار مع أو بدون -s، يجب عليك إجراء تنظيف إضافي للإخراج الذي تم التقاطه باستخدام re.sub(r'~.*\r', '', buf, 0, re.M).

ولكن بعد ذلك، لدينا مساعد سياق wrapper للتعامل مع كل ذلك تلقائيًا، بغض النظر عما إذا كان يحتوي على أحرف \r أم لا، لذلك فهو ببساطة:

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

هنا مثال اختبار كامل:

```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

إذا كنت تريد التقاط stderr، استخدم فئة CaptureStderr بدلاً من ذلك:

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

إذا كنت بحاجة إلى التقاط كلا التدفقين في نفس الوقت، استخدم فئة CaptureStd الأساسية:

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

أيضًا، للمساعدة في تصحيح مشكلات الاختبار، تقوم مديري السياق هؤلاء بشكل افتراضي بإعادة تشغيل التدفقات التي تم التقاطها تلقائيًا عند الخروج من السياق.

## التقاط تدفق سجل

إذا كنت بحاجة إلى التحقق من إخراج سجل معين، فيمكنك استخدام CaptureLogger:

```python
from transformers import logging
from transformers.testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

## الاختبار باستخدام متغيرات البيئة

إذا كنت تريد اختبار تأثير متغيرات البيئة لاختبار معين، فيمكنك استخدام الديكور المساعد mockenv في transformers.testing_utils:

```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

في بعض الأحيان، يحتاج برنامج خارجي إلى الاستدعاء، والذي يتطلب ضبط PYTHONPATH في os.environ لتضمين مسارات محلية متعددة. وهنا تأتي فئة المساعدة TestCasePlus في transformers.test_utils للمساعدة:

```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # الآن قم باستدعاء البرنامج الخارجي، ومرر إليه env
```

اعتمادًا على ما إذا كان ملف الاختبار موجودًا في مجموعة اختبارات tests أو examples، فسيقوم بإعداد env[PYTHONPATH] بشكل صحيح لتضمين أحد هذين الدليلين، وكذلك دليل src لضمان إجراء الاختبار على المستودع الحالي، وأخيرًا مع أي env[PYTHONPATH] الذي تم ضبطه بالفعل قبل استدعاء الاختبار إذا كان هناك أي شيء.

تُنشئ طريقة المساعدة هذه نسخة من كائن os.environ، لذلك يظل الكائن الأصلي سليمًا.

## الحصول على نتائج قابلة للتكرار

في بعض الحالات، قد ترغب في إزالة العشوائية من اختباراتك. للحصول على نتائج متطابقة وقابلة للتكرار، ستحتاج إلى تثبيت البذرة:

```python
seed = 42

# مولد الأرقام العشوائية في بايثون
import random

random.seed(seed)

# مولدات الأرقام العشوائية في باي تورش
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# مولد الأرقام العشوائية في نومبي
import numpy as np

np.random.seed(seed)

# مولد الأرقام العشوائية في تنسرفلو
tf.random.set_seed(seed)
```

## تصحيح الاختبارات

لبدء مصحح الأخطاء في نقطة التحذير، قم بما يلي:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## العمل مع سير عمل إجراءات جيت هاب

لتشغيل مهمة CI ذاتية الدفع، يجب عليك:

1. إنشاء فرع جديد على أصل transformers (ليس فرعًا!).
2. يجب أن يبدأ اسم الفرع بـ ci_ أو ci- (يؤدي main أيضًا إلى تشغيله، ولكن لا يمكننا إجراء PRs على main). يتم تشغيله أيضًا فقط لمسارات محددة - يمكنك العثور على التعريف المحدث في حالة تغييره منذ كتابة هذه الوثيقة [هنا](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml) تحت *دفع*.
3. قم بإنشاء طلب سحب من هذا الفرع.
4. بعد ذلك، يمكنك رؤية المهمة تظهر [هنا](https://github.com/huggingface/transformers/actions/workflows/self-push.yml). قد لا يتم تشغيله على الفور إذا كان هناك تراكم.
## تجربة ميزات CI التجريبية

يمكن أن تكون تجربة ميزات CI مشكلة محتملة لأنها قد تتعارض مع التشغيل الطبيعي لـ CI. لذلك، إذا كان يجب إضافة ميزة CI جديدة، فيجب القيام بذلك على النحو التالي.

1. قم بإنشاء مهمة مخصصة جديدة لاختبار ما يحتاج إلى اختبار.
2. يجب أن تنجح المهمة الجديدة دائمًا حتى تعطينا علامة ✓ خضراء (التفاصيل أدناه).
3. اتركها تعمل لبضعة أيام لترى أن مجموعة متنوعة من أنواع طلبات السحب يتم تشغيلها عليها (فروع مستودع المستخدم، والفروع غير المتفرعة، والفروع المنشأة من خلال تحرير ملف واجهة مستخدم github.com مباشرة، والدفعات القسرية المختلفة، إلخ - هناك الكثير) أثناء مراقبة سجلات الوظيفة التجريبية (ليس الوظيفة الكلية خضراء لأنه دائمًا ما يكون أخضرًا عن قصد).
4. عندما يكون من الواضح أن كل شيء صلب، قم بدمج التغييرات الجديدة في الوظائف الموجودة.

بهذه الطريقة، لن تتعارض التجارب على وظيفة CI نفسها مع سير العمل العادي.

والآن، كيف يمكننا جعل الوظيفة تنجح دائمًا أثناء تطوير ميزة CI الجديدة؟

تدعم بعض أنظمة CI، مثل TravisCI ignore-step-failure وستبلغ عن نجاح الوظيفة بشكل عام، ولكن CircleCI وGithub Actions لا تدعمان ذلك اعتبارًا من وقت كتابة هذا التقرير.

لذلك، يمكن استخدام الحل البديل التالي:

1. `set +euo pipefail` في بداية أمر التشغيل لقمع معظم حالات الفشل المحتملة في نص Bash البرمجي.
2. يجب أن يكون الأمر الأخير ناجحًا: `echo "done"` أو فقط `true` سيفعل ذلك

فيما يلي مثال:

```yaml
- run:
name: run CI experiment
command: |
set +euo pipefail
echo "setting run-all-despite-any-errors-mode"
this_command_will_fail
echo "but bash continues to run"
# emulate another failure
false
# but the last command must be a success
echo "during experiment do not remove: reporting success to CI, even if there were failures"
```

بالنسبة للأوامر البسيطة، يمكنك أيضًا القيام بما يلي:

```bash
cmd_that_may_fail || true
```

بالطبع، بمجرد الرضا عن النتائج، قم بدمج الخطوة التجريبية أو الوظيفة مع بقية الوظائف العادية، مع إزالة `set +euo pipefail` أو أي أشياء أخرى قد تكون أضفتها لضمان عدم تدخل الوظيفة التجريبية في التشغيل الطبيعي لـ CI.

كانت هذه العملية برمتها ستكون أسهل بكثير إذا كان بإمكاننا فقط تعيين شيء مثل `allow-failure` للخطوة التجريبية، والسماح لها بالفشل دون التأثير على الحالة العامة لطلبات السحب. ولكن، كما ذكر سابقًا، لا تدعم CircleCI وGithub Actions ذلك في الوقت الحالي.

يمكنك التصويت على هذه الميزة ومعرفة مكانها في هذه المواضيع الخاصة بـ CI:

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)

## تكامل DeepSpeed

بالنسبة لطلب سحب يتضمن تكامل DeepSpeed، ضع في اعتبارك أن إعداد CI الخاص بطلبات سحب CircleCI لا يحتوي على وحدات معالجة رسومية (GPU). يتم تشغيل الاختبارات التي تتطلب وحدات معالجة رسومية (GPU) على نظام تكامل مستمر مختلف ليليًا. وهذا يعني أنه إذا حصلت على تقرير CI ناجح في طلب السحب الخاص بك، فهذا لا يعني أن اختبارات DeepSpeed قد نجحت.

لتشغيل اختبارات DeepSpeed:

```bash
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

يتطلب إجراء أي تغييرات على تعليمات برمجة النماذج أو أمثلة PyTorch تشغيل اختبارات حديقة الحيوانات أيضًا.

```bash
RUN_SLOW=1 pytest tests/deepspeed
```