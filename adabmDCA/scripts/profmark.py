import argparse
import logging
import os
import torch
from adabmDCA.fasta import import_from_fasta, get_tokens, write_fasta
from adabmDCA.cobalt import run_cobalt
from adabmDCA.parser import add_args_profmark

# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description="Splits a multi-sequence alignment FASTA file into training and test sets using the Cobalt algorithm described in Petti et al (2022).")
    parser = add_args_profmark(parser)
    
    return parser

def main(args):
    print("\n" + "".join(["*"] * 10) + f" Splitting dataset into train and test sets " + "".join(["*"] * 10) + "\n")
    logging.basicConfig(level=logging.INFO)
    
    # Set the device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # check if the output directory exists
    if os.path.dirname(args.output_prefix):
        if not os.path.exists(os.path.dirname(args.output_prefix)):
            os.makedirs(os.path.dirname(args.output_prefix))
    # check if the input MSA exists
    if not os.path.exists(args.input_msa):
        raise FileNotFoundError(f"Input MSA file {args.input_msa} does not exist.")
    
    # load the MSA
    logging.info(f"Loading MSA from {args.input_msa}")
    tokens = get_tokens(args.alphabet)
    headers, msa = import_from_fasta(args.input_msa, tokens)
    msa = torch.tensor(msa, device=device)
    logging.info(f"Loaded {len(headers)} sequences of length {len(msa[0])} from {args.input_msa}")
    
    # run the Cobalt algorithm
    geom_mean = 0
    rnd_gen = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.bestof):
        logging.info(f"Running Cobalt algorithm iteration {i+1}/{args.bestof}")
        headers_train_prop, train_prop, headers_test_prop, test_prop = run_cobalt(
            headers,
            msa,
            args.t1,
            args.t2,
            args.t3,
            args.maxtrain,
            args.maxtest,
            rnd_gen,
        )
        geom_mean_prop = len(train_prop) * len(test_prop)
        logging.info(f"---> Iteration {i+1}: len(train)={len(train_prop)}, len(test)={len(test_prop)}")
        if geom_mean_prop > geom_mean:
            geom_mean = geom_mean_prop
            headers_train, train, headers_test, test = (
                headers_train_prop,
                train_prop,
                headers_test_prop,
                test_prop,
            )
    if geom_mean == 0:
        logging.warning("FAILED: No split was found for t1={}, t2={}, t3={}".format(args.t1, args.t2, args.t3))
        return
    logging.info(f"Best iteration: len(train)={len(train)}, len(test)={len(test)}, geom_mean={geom_mean}")
    # write the training and test sets to files
    write_fasta(
        os.path.join(args.output_prefix + ".train.fasta"),
        headers_train,
        train,
        tokens=tokens,
    )
    write_fasta(
        os.path.join(args.output_prefix + ".test.fasta"),
        headers_test,
        test,
        tokens=tokens,
    )
    logging.info(f"Training and test sets written in {os.path.dirname(args.output_prefix)}")
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)