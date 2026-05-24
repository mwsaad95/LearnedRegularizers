#!/bin/bash
echo "Starting the evaluation..."
python variational_reconstruction.py --problem Denoising --evaluation_mode IFT --regularizer_name CRR