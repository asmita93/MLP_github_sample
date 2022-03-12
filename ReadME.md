## Summary

This sample project is a snippet of Asmita Sinha's Master Thesis. 
It is aimed at training and validation of the RESMLP model on Imagenet-1k.

More information about RESMLP below
https://medium.com/@asmitasinha/resmlp-feedforward-networks-for-image-classification-with-data-efficient-training-e770df4d623f 

## Usage 
python resmlp_on_ImageNet.py --distributed=1  --ngpus='2' --mode="train"

## Libraries and Frameowrks 
It uses no frameworks for training and validation and is built only on pytorch. 
Imagenet is loaded via Datadings library(helps in faster processing with Parallel runs)
Training is done parallelly using DistributedDataParallel Library by Pytorch.
