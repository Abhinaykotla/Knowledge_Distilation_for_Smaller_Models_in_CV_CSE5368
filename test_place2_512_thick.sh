CUDA_VISIBLE_DEVICES=0 python bin/predict.py \
    model.path=$(pwd)/experiments/DiffIRS2-place/ \
    indir=$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
    outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt
