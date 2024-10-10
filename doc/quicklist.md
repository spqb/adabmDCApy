## Quicklist

- Train a bmDCA model with default arguments:
  ```bash
  $ ./adabmDCA.sh train -d <fasta_file> -o <output_folder>
  ```

- Restore the training of a bmDCA model:
  
  ```bash
   $ ./adabmDCA.sh train -d <fasta_file> -o <output_folder> -p <file_params> -c <file_chains>
  ```

- Train an eaDCA model with default arguments:
  ```bash
  $ ./adabmDCA.sh train -m eaDCA -d <fasta_file> -o <output_folder> --nsweeps 5
  ```

- Restore the training of an eaDCA model:
  ```bash
  $ ./adabmDCA.sh train -m eaDCA -d <fasta_file> -o <output_folder> -p <file_params> -c <file_chains>
  ```

- Decimate a bmDCA model at 2\% of density:
  ```bash
  ./adabmDCA.sh train -m edDCA -d <fasta_file> -p <file_params> -c <file_chains>
  ```

- Train and decimate a bmDCA model at 2\% of density:
  ```bash
  $ ./adabmDCA.sh train -m edDCA -d <fasta_file> 
  ```

- Sample from a previously trained DCA model:
  ```bash
  $ ./adabmDCA.sh sample -p <file_params> -d <fasta_file> -o <output_folder> --ngen <num_gen>
  ```

- Scoring a sequence set:
  ```bash
  $ ./adabmDCA.sh  energies -d <fasta_file>  -p <file_params>  -o <output_folder>
  ```

- Generating a single mutant library starting from a wild type:
  ```bash
  $ ./adabmDCA.sh DMS -d <WT> -p <file_params> -o <output_folder>
  ```

- Computing the matrix of Frobenius norms for the contact prediction:
  ```bash
  $ ./adabmDCA.sh contacts -p <file_params> -o <output_folder>
  ```