# Awesome-Earth-Artificial-Intelligence

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)  [![GitHub stars](https://img.shields.io/github/stars/ESIPFed/Awesome-Earth-Artificial-Intelligence)](https://github.com/ESIPFed/Awesome-Earth-Artificial-Intelligence/stargazers) [![Chat on slack](https://img.shields.io/badge/slack-join-ff69b4.svg)](https://esip-slack-invite.herokuapp.com/) [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FESIPFed%2FAwesome-Earth-Artificial-Intelligence)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FESIPFed%2FAwesome-Earth-Artificial-Intelligence)

A curated list of tutorials, notebooks, software, datasets, courses, books, video lectures and papers specifically for Artificial Intelligence (AI) use cases in Earth Science — with emphasis on open-source tools, freely accessible papers, and reproducible benchmarks (including geospatial and weather/climate foundation models).

Maintained by ESIP Machine Learning Cluster. Free and open to inspire AI for Good.

Contributions are most welcome. Please refer to our [contributing guidelines](contributing.md), [what is awesome?](awesome.md), and [Code of Conduct](code-of-conduct.md).

## Contents

| | | | | |
| - | - | - | - | - |
| [Courses](#courses) | [Books](#books) | [Tools](#tools) | [Foundation Models](#foundation-models) | [Tutorials](#tutorials) |
| [Training Datasets](#training-data) | [Code](#code) | [Videos](#videos) | [Papers](#papers) | [Reports](#reports) |
| [Thoughts](#thoughts) | [Competitions](#competitions) | [Communities](#communities) | [RelatedAwesome](#relatedawesome) | |



## ML-enthusiastic Earth Scientific Questions

| Earth Spheres | Scientific Problems |
| - | - |
| Geosphere | <ul><li>How to identify hidden signals of earthquakes?</li> <li>How to learn the spatio-temporal relationships amonog earthquakes and make predictions based on the relationship?</li> <li>How to capture complex relationships of volcano-seismic data and classify explosion quakes in volcanos?</li> <li>How to predict landslides</li> <li>How to estimate the damage?</li></ul> |
| Atmosphere | <ul><li>How to trace and predict climate change using machine learning?</li><li>How to predict hurricane?</li><li>How to monitor and predict meteorological drought?</li><li>How to detect wildfire early?</li><li>How to monitor and predict air quality?</li><li>How to predict dust storm?</li><li>How to accelerate the model simulation and lower the computing costs?</li></ul> |
| Hydrosphere | <ul><li>How to do high spatio-temporal resoluton waterbody mapping?</li><li>How to get insights of water quality from remote sensing?</li><li>How to monitor, and predict snow melt as a water resource?</li></ul> |
| Biosphere | <ul><li>How to do high spatio-temporal resoluton forest mapping?</li><li>How to do high spatio-temporal resoluton crop mapping?</li><li>How to do high spatio-temporal resoluton animal mapping?</li><li>How to fine-tune geospatial foundation models with sparse labels?</li></ul> |
| Cryosphere | <ul><li>How to do high spatio-temporal resoluton mapping and classification of sea ice?</li><li>How to monitor and predict glacier/ice sheet mass loss?</li></ul> |
| Cross-cutting | <ul><li>How to benchmark geospatial foundation models reproducibly across sensors and tasks?</li><li>How to combine physics-based models with machine learning for weather and climate?</li><li>How to build trustworthy, uncertainty-aware AI for operational Earth science?</li><li>How to use vision-language models for interactive Earth observation analysis?</li></ul> |


| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Courses

* :sunglasses::sparkling_heart: [GeoSMART Machine Learning Curriculum](https://geo-smart.github.io/curriculum)

* :sunglasses::sparkling_heart: [Introduction to Machine Learning for Earth Observation (EO College MOOC)](https://eo-college.org/courses/introduction-to-machine-learning-for-earth-observation/) - Free MOOC from TUM/DLR covering classification, object detection, change detection, SAR, and self-supervised learning for EO

* :sunglasses::sparkling_heart: [GeoAI with Python: A Practical Guide to Open-Source Geospatial AI](https://github.com/giswqs/GeoAI-Book) [Zenodo](https://zenodo.org/records/19207014) - Open-access book with 23 chapters of executable code for segmentation, detection, change detection, and foundation models

* [ICESat-2 Hackweek](https://icesat-2-2023.hackweek.io)

* [ML Seminar: Physics-informed Machine learning for weather and climate science](https://www.youtube.com/watch?v=B_4TONeY75U) (57:35) by Dr. Karthik Kashinath from Lawrence Berkeley National Lab, Mar 19, 2021

* [ML Seminar: Scalable Geospatial Analysis](https://www.youtube.com/watch?v=84VNWk_zFTM) (53:23) by Tom Augspurger from Microsoft AI for Earth, May 20, 2021 

* [Fundamentals of ML and DL in Python](https://github.com/ageron/handson-ml) - A series of Jupyter notebooks that walk you through the fundamentals of Machine Learning and Deep Learning in python using Scikit-Learn and TensorFlow. 

* [Trustworthy Artificial Intelligence for Environmental Science (TAI4ES) Summer School](https://www2.cisl.ucar.edu/tai4es) will be virtually the week of July 26-30, 2021.

* [Artificial Intelligence for Earth System Science (AI4ESS) Summer School](https://www2.cisl.ucar.edu/events/summer-school/ai4ess/2020/artificial-intelligence-earth-system-science-ai4ess-summer-school) [repo](https://github.com/NCAR/ai4ess-hackathon-2020) [readinglist](https://www2.cisl.ucar.edu/sites/default/files/AI4ESS%20Webpage%20PDF%20Recommended%20Readings.pdf)

* [American Meterological Survey AI Webinar Series](https://www.ametsoc.org/index.cfm/ams/webinar-directory/)

* [USGS Artificial Intelligence/Machine Learning Workshop](https://my.usgs.gov/confluence/pages/viewpage.action?pageId=613780355)

* [Stanford CS 229 ML Cheatsheets](https://github.com/afshinea/stanford-cs-229-machine-learning)

* [RadiantEarth ML4EO Bootcamp 2021](https://github.com/RadiantMLHub/ml4eo-bootcamp-2021)

* [Summer School on High-Performance and Disruptive Computing in Remote Sensing - Scaling Machine Learning for Remote Sensing using Cloud Computing](https://github.com/nasa-impact/workshop_notebooks)

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Books

* :sunglasses: :sparkling_heart: [Artificial Intelligence in Earth Science](https://www.google.com/books/edition/Artificial_Intelligence_in_Earth_Science/iH-HEAAAQBAJ?hl=en&gbpv=1&printsec=frontcover)

* :sunglasses: :sparkling_heart: [Artificial Intelligence Methods in the Environmental Sciences](https://books.google.com/books?id=0N4XBd5yl6oC&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false)

* [Deep Learning for the Earth Sciences](https://books.google.com/books?id=Wd3gzgEACAAJ&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false)

* [How to achieve AI maturity and why it matters? (PDF)](https://www.amdocs.com/sites/default/files/filefield_paths/ai-maturity-model-whitepaper.pdf)

* [70-Years-of-Machine-Learning-in-Geoscience-in-Review](https://github.com/JesperDramsch/70-Years-of-Machine-Learning-in-Geoscience-in-Review)

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Tools

* [GRIME AI](https://github.com/GRIME-Lab/GRIME-AI/wiki) [website](https://gaugecam.org/): A comprehensive suite of tools that facilitates a complete workflow for ecohydrological research using public and private collections of ground-based time-lapse imagery. From image acquisition to visualization to machine learning and artificial intelligence applications.,

* [GRIME2](https://github.com/gaugecam-dev/GRIME2/wiki) [website](https://gaugecam.org/): Software dedicated to measuring water level using a camera. Use ground-based time-lapse imagery with an octagon target in the scene and automatically measure water level.,

* [eo-learn](https://github.com/sentinel-hub/eo-learn): Earth observation processing framework for machine learning in Python,

* [EarthML](https://github.com/pyviz-topics/EarthML) [website](http://earthml.holoviz.org/):  Tools for working with machine learning in earth science, 

* [ML visualization tool](https://github.com/lutzroeder/netron) - A Visualization tool for neural network, deep learning and machine learning models, support ONNX (.onnx, .pb, .pbtxt), Keras (.h5, .keras), Core ML (.mlmodel), Caffe (.caffemodel, .prototxt), Caffe2 (predict_net.pb), Darknet (.cfg), MXNet (.model, -symbol.json), Barracuda (.nn), ncnn (.param), Tengine (.tmfile), TNN (.tnnproto), UFF (.uff) and TensorFlow Lite (.tflite). 

* [Dopamine](https://github.com/google/dopamine)  is a research framework for fast prototyping of reinforcement learning algorithms, 

* [mlflow](https://github.com/mlflow/mlflow) - MLflow: A Machine Learning Lifecycle Platform, 

* [Snips NLU](https://github.com/snipsco/snips-nlu) Snips NLU (Natural Language Understanding) is a Python library that allows to extract structured information from sentences written in natural language. 

* [MindsDB](https://github.com/mindsdb/mindsdb) - MindsDB is an Explainable AutoML framework for developers built on top of Pytorch. It enables you to build, train and test state of the art ML models in as simple as one line of code. 

* [TensorFlow Hub](https://github.com/tensorflow/hub) TensorFlow Hub is a repository of reusable assets for machine learning with TensorFlow. In particular, it provides pre-trained SavedModels that can be reused to solve new tasks with less training time and less training data. 

* [Polyaxon](https://github.com/polyaxon/polyaxon) - Polyaxon, a platform for building, training, and monitoring large scale deep learning applications. A Machine Learning Platform for Kubernetes. 

* [SynapseML](https://github.com/microsoft/SynapseML) - SynapseML (previously MMLSpark) is an open source library to simplify the creation of scalable machine learning pipelines. Microsoft Machine Learning for Apache Spark, 

* [TransmogrifAI](https://github.com/salesforce/TransmogrifAI) - TransmogrifAI (pronounced trăns-mŏgˈrə-fī) is an AutoML library written in Scala that runs on top of Apache Spark. It was developed with a focus on accelerating machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity, and reuse. 

* [Microsoft AI for Earth API Platform](https://github.com/microsoft/AIforEarth-API-Platform) - Microsoft AI for Earth API Platform is a distributed infrastructure designed to provide a secure, scalable, and customizable API hosting, designed to handle the needs of long-running/asynchronous machine learning model inference. It is based on Azure and Kubernetes. 

* [OneFlow](https://github.com/Oneflow-Inc/oneflow) - OneFlow is a performance-centered and open-source deep learning framework. 

* [ml.js](https://github.com/mljs/ml) - ml.js - Machine learning tools in JavaScript. 

* [BentoML](https://github.com/bentoml/BentoML) - BentoML is an open-source framework for high-performance ML model serving. 

* [flashflight:](https://github.com/facebookresearch/flashlight) - flashflight: A C++ standalone library for machine learning. 

* [Xarray-Beam](https://github.com/google/xarray-beam) - Python library for building Apache Beam pipelines with Xarray datasets.

* :sunglasses: [pygeoweaver](https://github.com/ESIPFed/pygeoweaver) - Python library for AI & geospatial workflow management, FAIRness, tangibility and productivity improvement

* :sunglasses: [TorchGeo](https://github.com/torchgeo/torchgeo) [docs](https://torchgeo.readthedocs.io/) - PyTorch domain library with 100+ geospatial datasets, spatial samplers, multispectral transforms, and pretrained backbones (Prithvi, DOFA, ScaleMAE)

* :sunglasses: [TerraTorch](https://github.com/IBM/terratorch) [paper](https://arxiv.org/abs/2503.20563) - Fine-tuning and benchmarking toolkit for geospatial foundation models; integrates with GEO-Bench-2 and Hugging Face weights

* :sunglasses: [GeoAI](https://github.com/opengeos/geoai) [docs](https://opengeoai.org/) - Unified Python framework for EO deep learning: segmentation, detection, change detection, and foundation model workflows

* [segment-geospatial (samgeo)](https://github.com/opengeos/segment-geospatial) [docs](https://samgeo.gishub.org/) - Segment Anything Model (SAM) and HQ-SAM for geospatial imagery segmentation

* [WeatherBench 2](https://github.com/google-research/weatherbench2) [docs](https://weatherbench2.readthedocs.io/) - Open evaluation framework and leaderboard for data-driven global weather models

* [GEO-Bench-2](https://github.com/The-AI-Alliance/GEO-Bench-2) [leaderboard](https://huggingface.co/spaces/aialliance/GEO-Bench-2-Leaderboard) [paper](https://arxiv.org/abs/2511.15658) - Reproducible benchmark for geospatial foundation models across 19 permissively licensed datasets

* [Makani](https://github.com/NVIDIA/makani) - Scalable training framework for ML weather models (FourCastNet 3); Apache 2.0

* [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) - Differentiable signal processing on the sphere for geometric weather ML; BSD-3-Clause

* [SeisBench](https://github.com/seisbench/seisbench) [docs](https://seisbench.readthedocs.io/) - Open toolbox for earthquake ML: phase picking, event detection, pretrained models, and benchmark datasets

* [ClimateLearn](https://github.com/aditya-grover/climate-learn) [paper](https://arxiv.org/abs/2307.01909) - PyTorch library for weather forecasting and climate downscaling benchmarks (ERA5, CMIP6)

* [ai-models](https://github.com/ecmwf-lab/ai-models) - Open-source CLI to run AI weather models (GraphCast, FourCastNet, Pangu-Weather) with ECMWF data pipelines

* [Wildlife Insights](https://www.wildlifeinsights.org/) - Google-backed production platform for automated species identification from camera trap images using deep learning, integrating with GBIF and supporting large-scale biodiversity monitoring.

* [iNaturalist Computer Vision](https://www.inaturalist.org/pages/computer_vision_demo) - Species identification model trained on community-contributed observations covering 76,000+ taxa; powers the iNaturalist app's AI suggestion feature and is openly described in peer-reviewed literature.

* [Global Forest Watch](https://www.globalforestwatch.org/) - ML-powered deforestation and forest cover change monitoring platform using satellite imagery analysis; provides near-real-time alerts used by researchers and conservation organizations globally.

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Foundation Models

Open pretrained models for Earth observation, weather, and climate. We list models with publicly released weights and permissive open-source licenses.

### Earth Observation

* :sunglasses::sparkling_heart: [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) [weights](https://huggingface.co/ibm-nasa-geospatial) [paper](https://arxiv.org/abs/2412.02732) - Multi-temporal ViT foundation model (300M/600M) trained on 4.2M global HLS time series at 30 m

* :sunglasses::sparkling_heart: [Clay](https://github.com/Clay-foundation/model) [docs](https://clay-foundation.github.io/model/) [weights](https://huggingface.co/made-with-clay/Clay) - Sensor-agnostic MAE foundation model (v1.5) for EO embeddings across Sentinel-2, Landsat, Sentinel-1, and custom sensors; Apache 2.0

* [TerraMind](https://github.com/ibm/terramind) [weights](https://huggingface.co/ibm-esa-geospatial) [paper](https://arxiv.org/abs/2504.11171) - Any-to-any generative multimodal EO foundation model (IBM/ESA Φ-lab); integrates with TerraTorch

* [DOFA](https://github.com/zhu-xlab/DOFA) [paper](https://arxiv.org/abs/2403.15356) - Dynamic One-For-All multimodal foundation model with wavelength-conditioned hypernetworks for cross-sensor generalization

* [Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM) [paper](https://arxiv.org/abs/2503.11849) - Unified Copernicus foundation model across Sentinel missions with Copernicus-Pretrain and Copernicus-Bench

* [AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) [embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) - Global 10 m embedding field layers (2017–2024) for sparse-label mapping; annual embeddings on Google Earth Engine and GCS

### Weather and Climate

* :sunglasses::sparkling_heart: [Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC) [weights](https://huggingface.co/Prithvi-WxC) [paper](https://arxiv.org/abs/2409.13598) - 2.3B-parameter weather/climate foundation model on MERRA-2 for forecasting, downscaling, and parameterization

* [GraphCast / GenCast](https://github.com/google-deepmind/graphcast) [GraphCast paper](https://arxiv.org/abs/2212.12794) [GenCast paper](https://arxiv.org/abs/2312.15796) - GNN-based medium-range global weather forecasting and diffusion ensemble forecasting; Apache 2.0

* [Aurora](https://github.com/microsoft/aurora) [docs](https://microsoft.github.io/aurora/) [paper](https://arxiv.org/abs/2405.13063) - 1.3B-parameter atmospheric foundation model for weather, air pollution, and ocean waves

* [NeuralGCM](https://github.com/neuralgcm/neuralgcm) [dycore](https://github.com/neuralgcm/dinosaur) [paper](https://arxiv.org/abs/2311.07222) - Differentiable hybrid general circulation model combining physics-based dynamics with learned components; Apache 2.0 code, CC BY-SA 4.0 weights

* [FourCastNet 3](https://arxiv.org/abs/2507.12144) [training code](https://github.com/NVIDIA/makani) - Probabilistic spherical-convolution weather ensemble forecasting at 0.25°; open training via Makani

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Tutorials

* :sunglasses::sparkling_heart: [GeoSMART Machine Learning Curriculum & Use Cases](https://geo-smart.github.io/usecases)

* :sunglasses::sparkling_heart: [GeoAI with Python Book Code](https://github.com/giswqs/GeoAI-Book) - Executable notebooks for seven core GeoAI tasks and foundation model workflows

* :sunglasses::sparkling_heart: [GeoAI Video Tutorials](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE) [docs](https://opengeoai.org/) - Step-by-step GeoAI package tutorials from Open Geospatial Solutions

* :sunglasses::sparkling_heart: [TerraTorch Documentation](https://terrastackai.github.io/terratorch/stable/) - Fine-tuning guides for Prithvi, TerraMind, Clay, and GEO-Bench-2 benchmarking

* :sunglasses::sparkling_heart: [NeuralGCM Inference Quickstart](https://neuralgcm.readthedocs.io/en/latest/inference_demo.html) - Run pretrained hybrid GCM weather forecasts with open checkpoints on GCS

* :sunglasses::sparkling_heart: [NASA Openscapes Earthdata Cloud Cookbook](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/our-cookbook.html)

* :sunglasses::sparkling_heart: [Artificial Intelligence in Earth science Book Materials](https://github.com/earth-artificial-intelligence/earth_ai_book_materials)

* :sunglasses::sparkling_heart: [RadiantEarth MLhub Tutorials](https://github.com/radiantearth/mlhub-tutorials)

* [Machine Learning Tutorials (general, not Earth science specific)](https://github.com/ethen8181/machine-learning)

* [Pixel-level land cover classification](https://github.com/Azure/pixel_level_land_classification)

* [EO-learn-workshop](https://github.com/sentinel-hub/eo-learn-workshop) - EO-learn-workshop: Bridging Earth Observation data and Machine Learning in Python, 

* [Machine Learning for Development](https://github.com/worldbank/ml4dev) Machine Learning for Development: A method to Learn and Identify Earth Features from Satellite Images, 

* [ELSI-DL-Bootcamp](https://github.com/Machine-Learning-Tokyo/ELSI-DL-Bootcamp) - Intro to Machine Learning and Deep Learning for Earth-Life Sciences, 

* [UW WaterhackerWeek](https://github.com/waterhackweek/whw2020_machine_learning) - Introduction to Machine Learning on Landslide Data and Scikit-learn from [UW WaterhackerWeek](https://waterhackweek.github.io/), 

* [Planet Snow Mapping](https://github.com/acannistra/planet-snowcover) - Introduction to using Planet imagery to map snow cover

* [Machine Learning Pipeline for Climate Science](https://ml-clim.github.io/drought-prediction/) - an end-to-end pipeline for the creation, intercomparison and evaluation of machine learning methods in climate science

* [AI Cheatsheets](https://github.com/kailashahirwar/cheatsheets-ai) - Essential Cheat Sheets for deep learning and machine learning engineers. It contains a lot of useful tutorials to learn awesome tricks on AI engineering

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Training Data

* [Kaggle Earth Science Training Dataset](https://www.kaggle.com/search?q=tag%3A%22earth+science%22+in%3Adatasets)

* [Radiant MLHub](https://www.mlhub.earth/#datasets) - an open library for geospatial training data

* [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog)

* [AlphaEarth Satellite Embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) [paper](https://arxiv.org/abs/2507.22291) - Global annual 10 m embedding fields (2017–2024) from AlphaEarth Foundations; also on [GCS](https://console.cloud.google.com/storage/browser/alphaearth_foundations)

* [GEO-Bench-2 Datasets](https://github.com/The-AI-Alliance/GEO-Bench-2) - 19 permissively licensed benchmark datasets for geospatial foundation model evaluation on Hugging Face

* [Copernicus-Embed-025deg](https://github.com/zhu-xlab/Copernicus-FM) - Global 0.25° embedding map integrating multi-source Sentinel observations (released with Copernicus-FM)

* [WeatherBench 2 ERA5 Zarr](https://github.com/google-research/weatherbench2) - Open cloud-optimized ERA5 and baseline forecast data for ML weather model training and evaluation

* [University of California Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

* [EuroSAT Dataset](https://github.com/phelber/EuroSAT) - EuroSAT Dataset: Land Use and Land Cover Classification with Sentinel-2, 

* [Awesome Satellite Imagery Datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) - Awesome Satellite Imagery Datasets: A curated list of deep learning training datasets, 

* [STanford EArthquake Dataset (STEAD)](https://github.com/smousavi05/STEAD) - A Global Data Set of Seismic Signals for AI

* [ZipCheckup](https://zipcheckup.com) - Free ZIP-level environmental safety dataset for 42,000+ US ZIP codes covering water quality, air quality, PFAS contamination, radon, lead, flood risk, and 11 additional verticals. Public REST API and npm/PyPI packages for ML pipelines. CC BY 4.0.

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Code

* :sunglasses::sparkling_heart: [Earth System Emulator (ESEm)](https://github.com/duncanwp/ESEm) - A tool for emulating geophysical datasets including (but not limited to) Earth System Models

* :sunglasses::sparkling_heart: [EmissionAI](https://github.com/ZihengSun/EmissionAI) - Microsoft AI for Earth Project: AI Monitoring Coal-fired Power Plant Emission from Space

* [Prithvi-EO-2.0 Fine-tuning Examples](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) - TerraTorch configs for burn scars, flood mapping, crop type, and multi-temporal land cover

* [TEOChat](https://github.com/ermongroup/TEOChat) [paper](https://arxiv.org/abs/2410.06234) - Temporal vision-language assistant for change detection, damage assessment, and EO dialogue

* [GeoChat](https://github.com/mbzuai-oryx/GeoChat) [paper](https://arxiv.org/abs/2311.15826) - Grounded large vision-language model for remote sensing QA, captioning, and referring detection

* [EarthDial](https://github.com/hiyamdebary/EarthDial) [paper](https://arxiv.org/abs/2412.15190) - Multi-spectral, multi-temporal vision-language model for EO dialogue across 44 downstream datasets

* [Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM) - Unified Copernicus foundation model with pretraining and benchmark code (ICCV 2025)

* [BassNet](https://github.com/hbutsuak95/BASS-Net),[paper-preprint](https://arxiv.org/abs/1612.00144) - Deep Learning for Land-cover Classification in Hyperspectral Images, 

* [MTLCC](https://github.com/TUM-LMF/MTLCC) - Multitemporal Land Cover Classification Network (ConvLSTM, ConvGRU), 

* [Landsat Time Series Analysis for Multi-Temporal Land Cover Classification](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)

* [EarthEngine-Deep-Learning](https://github.com/ucalyptus/EarthEngine-Deep-Learning) - Deep Learning on Google Earth Engine, 

* [Continuous Change Detection and Classification](https://github.com/GERSL/CCDC) - Continuous Change Detection and Classification (CCDC) of land cover using all available Landsat data, 

* [Object-based Classification on Earth Engine](https://github.com/GERSL/CCDC) - Object-based land cover classification with Feature Extraction and Feature Selection for Google Earth Engine (GEE), 

* [Earth Lens](https://github.com/microsoft/Earth-Lens) - Earth Lens, a Microsoft Garage project is an iOS iPad application that helps people and organizations quickly identify and classify objects in aerial imagery through the power of machine learning. 

* [Image Classification Neural Network Ranking with source code](https://paperswithcode.com/task/image-classification) - paperswithcode has put together a list of cutting-edge papers and ranked them with the claimed accuracy.

* [EQTransformer](https://github.com/smousavi05/EQTransformer) - An AI-Based Earthquake Signal Detector and Phase Picker. 

* [Tropical Cyclone Windspeed Estimator](https://github.com/drivendataorg/wind-dependent-variables) - Winning solutions for Tropical Cyclone Wind Speed Prediction Competition

* [Infernis](https://github.com/argonBIsystems/infernis) - Open-source ML-powered wildfire risk prediction engine for British Columbia. XGBoost + CNN trained on 10 fire seasons (2015-2024) from 21 open government and scientific data sources. Provides a free [REST API](https://infernis.ca/v1/docs) with daily predictions at 5km resolution.


| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Videos

* [GeoAI Tutorials Playlist](https://www.youtube.com/playlist?list=PLAxJ4-o7ZoPcvENqwaPa_QwbbkZ5sctZE) - Open Geospatial Solutions tutorials on segmentation, detection, and QGIS GeoAI plugin workflows

* [Tutorial on Microsoft Azure Machine Learning Studio (AutoML-Regression)](https://www.youtube.com/watch?v=ip5GHTMZPhA), created by Microsoft AI for Earth Project: AI Monitoring Coal-fired Power Plant Emission from Space.

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Papers

* :sunglasses: :sparkling_heart: [A Review of Earth Artificial Intelligence](https://www.sciencedirect.com/science/article/pii/S0098300422000036)

* :sunglasses: :sparkling_heart: [Foundation Models for Remote Sensing and Earth Observation: A Survey](https://arxiv.org/abs/2410.16602) - Taxonomy of visual, vision-language, and LLM-based RSFMs with benchmarking across public datasets

* [Towards practical artificial intelligence in Earth sciences](https://link.springer.com/article/10.1007/s10596-024-10317-7)

* [A Review of Practical AI for Remote Sensing in Earth Sciences](https://www.mdpi.com/2072-4292/15/16/4112)

* [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://arxiv.org/abs/2412.02732)

* [Prithvi WxC: Foundation Model for Weather and Climate](https://arxiv.org/abs/2409.13598)

* [TerraMind: Large-Scale Generative Multimodality for Earth Observation](https://arxiv.org/abs/2504.11171)

* [TerraTorch: The Geospatial Foundation Models Toolkit](https://arxiv.org/abs/2503.20563)

* [AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data](https://arxiv.org/abs/2507.22291)

* [GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI](https://arxiv.org/abs/2511.15658)

* [Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities (DOFA)](https://arxiv.org/abs/2403.15356)

* [FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale](https://arxiv.org/abs/2507.12144)

* [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)

* [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)

* [Aurora: A Foundation Model of the Atmosphere](https://arxiv.org/abs/2405.13063)

* [Neural General Circulation Models for Weather and Climate](https://arxiv.org/abs/2311.07222)

* [TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://arxiv.org/abs/2410.06234)

* [GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://arxiv.org/abs/2311.15826)

* [EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](https://arxiv.org/abs/2412.15190)

* [Towards a Unified Copernicus Foundation Model for Earth Vision](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_a_Unified_Copernicus_Foundation_Model_for_Earth_Vision_ICCV_2025_paper.html) [arxiv](https://arxiv.org/abs/2503.11849)

* [Advances on Multimodal Remote Sensing Foundation Models for Earth Observation Downstream Tasks: A Survey](https://www.mdpi.com/2072-4292/17/21/3532) - Open-access review of vision-X multimodal RSFMs

* [Big Earth data analytics: A survey](https://www.tandfonline.com/doi/full/10.1080/20964471.2019.1611175)

* [Adoption of machine learning techniques in ecology and earth science](https://oneecosystem.pensoft.net/article/8621/download/pdf/)

* [CIRA Guide To Custom Loss Functions For Neural Networks In Environmental Sciences - Version 1](https://arxiv.org/pdf/2106.09757.pdf)

* [Zero-Shot Learning of Aerosol Optical Properties with Graph NeuralNetworks](https://arxiv.org/pdf/2107.10197.pdf)

* [NeuralHydrology - a collection of papers on using neural networks in hydrology](https://neuralhydrology.github.io/)

* [Ten Ways to Apply Machine Learning in Earth and Space Sciences](https://eos.org/opinions/ten-ways-to-apply-machine-learning-in-earth-and-space-sciences)

* [Advancing AI for Earth Science: A Data Systems Perspective](https://eos.org/science-updates/advancing-ai-for-earth-science-a-data-systems-perspective)

* [Google Earth Engine: Planetary-scale geospatial analysis for everyone](https://www.sciencedirect.com/science/article/pii/S0034425717302900)

* [WeatherBench 2: A benchmark for the next generation of data-driven global weather models](https://arxiv.org/abs/2308.15560)

* [ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling](https://arxiv.org/abs/2307.01909)

* [PCA-OS: A Planetary Climate Adaptation Operating System](https://chaoyue0307.github.io/PCA-OS/) (KDD 2026 Blue Sky Ideas Track) - Frames climate adaptation as a continual learning and decision loop over an intervention-aware global causal knowledge graph, fusing Earth-observation signals and operational traces into versioned, auditable adaptation interventions and robust decision portfolios.


| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Reports

* [Workshop Report: Advancing Application of Machine Learning Tools for NASA’s Earth Observation Data](https://cdn.earthdata.nasa.gov/conduit/upload/14287/NASA_ML_Workshop_Report.pdf) 

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Thoughts

* :sunglasses: :sparkling_heart: [Learning earth system models from observations: machine learning or data assimilation?](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0089)

* [Artificial intelligence: A powerful paradigm for scientific research](https://www.sciencedirect.com/science/article/pii/S2666675821001041)

* [Why 90% of machine learning models never hit the market](https://thenextweb.com/news/why-most-machine-learning-models-never-hit-market-syndication)

* ['Farewell Convolutions' – ML Community Applauds Anonymous ICLR 2021 Paper That Uses Transformers for Image Recognition at Scale](https://syncedreview.com/2020/10/08/farewell-convolutions-ml-community-applauds-anonymous-iclr-2021-paper-that-uses-transformers-for-image-recognition-at-scale/)

* [37 reasons why your neural network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Competitions

* :sunglasses::sparkling_heart: [GeoAI Challenge](https://aiforgood.itu.int/about-ai-for-good/geoai-challenge/) - aimed at providing solutions for collaboratively addressing real-world geospatial problems by applying artificial intelligence (AI)/machine learning (ML)

* [2025 GeoAI Challenge: Cropland Mapping in Dry Environments](https://zindi.africa/competitions/geoai-challenge-for-cropland-mapping-in-dry-environments) - ITU/FAO challenge on distinguishing cropland from pasture in Fergana and Orenburg using time-series satellite imagery

* [2026 GeoAI Challenge: Reaching new heights with GeoFM](https://aiforgood.itu.int/about-us/geoai-challenge/) - ITU/ESA Φ-lab challenge on global surface height and land-cover mapping with open satellite imagery and GeoFM embeddings

* [GPU Hackthons](https://www.gpuhackathons.org/) - designed to help scientists, researchers and developers to accelerate and optimize their applications on GPUs.

* [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction)

* [HackerEarth](https://www.hackerearth.com/challenges/)



| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## Communities

* [ESIP Machine Learning Cluster](https://wiki.esipfed.org/Machine_Learning)

* [ESIP Agriculture and Climate Cluster](https://wiki.esipfed.org/Agriculture_and_Climate)

* [AI Alliance Climate & Sustainability Group](https://thealliance.ai/blog/geo-bench-2-from-performance-to-capability-rethinking-evaluation-in-geospatial-ai) - Community behind GEO-Bench-2 and open geospatial foundation model evaluation

* [TorchGeo Community](https://torchgeo.org/) - OSGeo community project for geospatial deep learning in PyTorch

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

## RelatedAwesome
- [Awesome-Open-Geoscience](https://github.com/softwareunderground/awesome-open-geoscience) – ![Awesome](media/icon/awesome.png) A list is curated from repositories that make our lives as geoscientists, hackers and data wranglers easier or just more awesome. In accordance with the awesome manifesto, we add awesome repositories.
- [Awesome-Spatial](https://github.com/RoboDonut/awesome-spatial) – ![Awesome](media/icon/awesome.png) Awesome list for geospatial, not specific to geoscience but significant overlap
- [Awesome Open Climate Science](https://github.com/pangeo-data/awesome-open-climate-science) – ![Awesome](media/icon/awesome.png) Awesome list for atmospheric, ocean, climate, and hydrologic science
- [awesome-weather-models](https://github.com/rebase-energy/awesome-weather-models) – ![Awesome](media/icon/awesome.png) Catalogue of AI-based weather forecasting models with open-source and open-weights status
- [awesome-WeatherAI](https://github.com/HeQinWill/awesome-WeatherAI) – ![Awesome](media/icon/awesome.png) Papers, datasets, and open model implementations for AI weather and climate
- [Awesome_AI4Earth](https://github.com/taohan10200/Awesome_AI4Earth) – ![Awesome](media/icon/awesome.png) Deep learning for Earth system science, especially data-driven weather prediction
- [Awesome-AI-for-Atmosphere-and-Ocean](https://github.com/XiongWeiTHU/Awesome-AI-for-Atmosphere-and-Ocean) – ![Awesome](media/icon/awesome.png) Research papers on AI for atmospheric science and oceanography
- [Awesome Coastal](https://github.com/chrisleaman/awesome-coastal) – ![Awesome](media/icon/awesome.png) Awesome list for coastal engineers and scientists
- [Awesome Satellite Imagery Datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) - ![Awesome](media/icon/awesome.png) List of aerial and satellite imagery datasets with annotations for computer vision and deep learning
- [Awesome Workflow Engines](https://github.com/meirwah/awesome-workflow-engines) - ![Awesome](media/icon/awesome.png) A curated list of awesome open source workflow engines
- [Awesome Pipeline](https://github.com/pditommaso/awesome-pipeline) - ![Awesome](media/icon/awesome.png) A curated list of awesome pipeline toolkits inspired by Awesome Sysadmin
- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) - ![Awesome](media/icon/awesome.png) A curated list of awesome Machine Learning frameworks, libraries and software

| ▲ [Top](#awesome-earth-artificial-intelligence) |
| --- |

  

