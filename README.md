# azureml-designer-demo

#### File Description
1. split_data_txt.py, fasttext_train.py and fasttext_test.py are source codes of Split Data Txt module, FastText Train module and FastText Test module respectively in designer.

2. sample_fasttext_pipeline.ipynb shows one training and testing process of fastText

3. sample_fasttext_pipeline2.ipynb shows two training and testing processes of fastText

#### Environment Configuration
1. Create codespace on [Visual Studio Codespaces](https://visualstudio.microsoft.com/services/visual-studio-codespaces/). Click "Get started"
    <img src="https://note.youdao.com/yws/public/resource/d68209b5672655918654069ad86b7ac0/xmlnote/9E0278990A6941649D18519D6B42D1AD/72068" style="zoom:50%;" />

2. Click "Create Codespace". Then fill in the options . Here, you need to input ```https://github.com/smallhaes/azureml-designer-demo.git``` in the "Git Repository" box

   <img src="https://note.youdao.com/yws/public/resource/d68209b5672655918654069ad86b7ac0/xmlnote/EB35D0C2D8C849AC832EB3658C43880F/72070" style="zoom: 67%;" />

3. After entering our codespace, change directory to ```~/workspace/azureml-designer-demo``` and run ```sh prepare.sh ```. It will about 5 minutes to configure the environment.
4. Open ```sample_fasttext2_pipeline.ipynb``` and change the kernel to ```Python 3.7.5 64-bit('pythonenv3.7':venv)```
![](https://note.youdao.com/yws/public/resource/d68209b5672655918654069ad86b7ac0/xmlnote/AD8766471B274486946271544EFA8800/72082)

5. Now you could run this ipynb file. When executing the second cell for the first time, you need to perform interactive authentication or it would be stuck here.![](https://note.youdao.com/yws/public/resource/d68209b5672655918654069ad86b7ac0/xmlnote/5FBEF7ED3BFE432F8575D128C6A9A4FA/72077)