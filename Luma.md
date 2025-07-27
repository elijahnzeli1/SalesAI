Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Pricing



Datasets:

bezirganyan
/
LUMA 

like
3
Tasks:
Image Classification
Audio Classification
Text Classification
Modalities:
Audio
Formats:
soundfolder
Languages:
English
Size:
1K - 10K
ArXiv:

arxiv:
2406.09864
Tags:
uncertainty quantification
multimodal classification
multimodal uncertainty classification
DOI:

doi:10.57967/hf/2502
Libraries:
Datasets

Croissant
License:

cc-by-sa-4.0
Dataset card
Data Studio
Files and versions
Community
176
LUMA
/
README.md

bezirganyan's picture
bezirganyan
Update README.md
8fac10c
verified
24 days ago
preview
code
|
raw

Copy download link
history
blame
contribute
delete

9.92 kB
---
license: cc-by-sa-4.0
task_categories:
- image-classification
- audio-classification
- text-classification
language:
- en
tags:
- uncertainty quantification
- multimodal classification
- multimodal uncertainty classification
pretty_name: 'LUMA: Learning from Uncertain and Multimodal Data'
size_categories:
- 100K<n<1M
modalities:
- image
- audio
- text
---
<!-- # LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data -->

<!-- Provide a quick summary of the dataset. -->
<div style="text-align: center; background: linear-gradient(to right, #001f3f, #0074D9); padding: 20px; border-radius: 10px; color: white;">
    <h1 style="font-size: 3em; margin: 0; color: white;">LUMA</h1>
    <p style="font-size: 1.5em; margin: 0;">A Benchmark Dataset for Learning from Uncertain and Multimodal Data</p>
    <div style="margin: 20px 0;">
        <span style="font-size: 2em; margin: 0 10px;">📄</span>
        <span style="font-size: 2em; margin: 0 10px;">📷</span>
        <span style="font-size: 2em; margin: 0 10px;">🎵</span>
        <span style="font-size: 2em; margin: 0 10px;">📊</span>
        <span style="font-size: 2em; margin: 0 10px;">❓</span>
    </div>
    <p style="font-style: italic; font-size: 1.2em; margin: 0;">Multimodal Uncertainty Quantification at Your Fingertips</p>
</div>
The LUMA dataset is a multimodal dataset, including audio, text, and image modalities, intended for benchmarking multimodal learning and multimodal uncertainty quantification.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
LUMA is a multimodal dataset that consists of audio, image, and text modalities. It allows controlled injection of uncertainties into the data and is mainly intended for studying uncertainty quantification in multimodal classification settings. 
This repository provides the Audio and Text modalities. The image modality consists of images from [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. 
To download the image modality and compile the dataset with a specified amount of uncertainties, please use the [LUMA compilation tool](https://github.com/bezirganyan/LUMA). 

<!-- - **Curated by:** [More Information Needed] -->
<!-- - **Funded by [optional]:** [More Information Needed] -->
<!-- - **Shared by [optional]:** [More Information Needed] -->
- **Language(s) (NLP):** English
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Paper:** ([preprint](https://arxiv.org/abs/2406.09864)) - Accepted to SIGIR 2025 conference

<!-- ### Dataset Sources
 -->
<!-- Provide the basic links for the dataset. -->

<!-- - **Repository:** [More Information Needed] -->
<!-- - **Demo [optional]:** [More Information Needed] -->

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

The dataset is intended to be used for studying and benchmarking multimodal classification. Researchers can use the provided Python tool to compile different versions of the datasets with different amounts of uncertainties. 

### Out-of-Scope Use

The dataset shall not be used as a source of knowledge or information. The text modality is generated using large-language models and can contain biases or factually incorrect information. 
<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
The dataset consists of audio, text, and image modalities.
**Image modality**: Image modality contains images from a 50-class subset from CIFAR-10/100 datasets, as well as generated images from the same distribution. 
**Audio modality**: Audio modality contains `wav` files of people pronouncing the class labels of the selected 50 classes.
**Text modality**: Text modality contains short text passages about the class labels, generated using large language models. 

The [provided Python tool](https://github.com/bezirganyan/LUMA) allows compiling different versions of the dataset, with different amounts and types of uncertainties. Each version of the dataset contains 42 classes, with 500 samples per class for training, and 100 samples per class for testing. The remaining 8 classes are provided as out-of-distribution (OOD) data. 

In the `audio` directory, we have the `datalist.csv`, with columns:
* `path`: the path of the related audio wav file
* `label`: label of the audio (the word that is being pronounced in the audio)
* `tts_label`: the label that is predicted by the Text-To-Speech (TTS) model

In the `audio`, the different directories contain audio files from different sources. 
* The `cv_audio` directory contains audio files from the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset. This dataset has [CC0](https://creativecommons.org/public-domain/cc0/) license, as described in their [release blog post](https://blog.mozilla.org/en/mozilla/news/sharing-our-common-voices-mozilla-releases-the-largest-to-date-public-domain-transcribed-voice-dataset/).
* The `sw_audio` directory contains audio files from the [The Spoken Wikipedia](https://nats.gitlab.io/swc/) dataset. This dataset has [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
* The `ls_audio` directory contains audio files from the [LibriSpeech](https://www.openslr.org/12) dataset. This dataset has [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
* The `re_audio` directory contains audio files recorded by us, from volunteered colleagues. These audio files, as well as the entire dataset, are shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The `text_data.tsv` file is a tab-separated file of text passages generated using the [Gemma 7B](https://huggingface.co/google/gemma-7b-it) Large Language Model (LLM). 
The column `text` contains the text passages, and the column `label` contains the labels of these texts. 

The `edm_images.pickle` is a pandas dataframe saved as a pickle, containing EDM generated images and their labels. It is retrieved from [DM-Improves-AT](https://huggingface.co/datasets/P2333/DM-Improves-AT) page, where it is published under the [Apache-2.0](https://apache.org/licenses/LICENSE-2.0) license. 

## Dataset Creation

### Curation Rationale

Building trustworthy multimodal models requires quantifying uncertainty in both the data and the model itself. Existing multimodal datasets lack the ability to controllably inject various types and amounts of uncertainty, such as data diversity, label noise, sample noise, and out-of-distribution (OOD) data. To address this limitation, we introduce the LUMA dataset, specifically designed to enable researchers to conduct controlled experiments in Multimodal Uncertainty Quantification (MUQ).

### Source Data

The audio data is word pronunciations extracted from the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets), [The Spoken Wikipedia](https://nats.gitlab.io/swc/), and [LibriSpeech](https://www.openslr.org/12) datasets. 

The text modality consists of short text passages generated using the [Gemma 7B](https://huggingface.co/google/gemma-7b-it).

The image modalities consist of CIFAR-10/100 datasets (need to be downloaded separately), and images generated from the same distribution. 
<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

<!-- #### Data Collection and Processing -->

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

<!-- [More Information Needed] -->

<!-- #### Who are the source data producers? -->

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

#### Personal and Sensitive Information

The dataset does not contain personal or sensitive information.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The text modality is generated using large language models (LLMs), hence it can contain biases or factually incorrect information. The use of the dataset shall be limited to studying multimodal uncertainty quantification, and shall not be used as a source of knowledge. 

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

The use of the dataset shall be limited to studying multimodal uncertainty quantification, and shall not be used as a source of knowledge. 

**BibTeX:**

```
@inproceedings{luma_dataset2025,
  title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data}, 
  author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

**APA:**

```
Bezirganyan, G., Sellami, S., Berti-Équille, L., & Fournier, S. (2025). LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval.
```

## Contact

* <a href="mailto:grigor.bezirganyan98@gmail.com">Grigor Bezirganyan</a>
* <a href="mailto:sana.sellami@univ-amu.fr">Sana Sellami</a>
* <a href="mailto:laure.berti@ird.fr">Laure Berti-Équille</a>
* <a href="mailto:sebastien.fournier@univ-amu.fr">Sébastien Fournier</a>
