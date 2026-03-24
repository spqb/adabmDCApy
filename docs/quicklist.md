
## <span id="quicklist">List of the main routines with standard arguments</span>

- **Train a `bmDCA` model** with default arguments:

```bash
adabmDCA train -d <fasta_file> -o <output_folder>
```
 
 - __Resume training__ of a `bmDCA` model:
 
```bash
adabmDCA train -d <fasta_file> -o <output_folder> -p <file_params> -c <file_chains>
```
  
- __Train an `eaDCA` model__ with default arguments:

```bash
adabmDCA train -m eaDCA -d <fasta_file> -o <output_folder> --nsweeps 5
```

- __Resume training__ of an eaDCA model:

```bash
adabmDCA train -m eaDCA -d <fasta_file> -o <output_folder> -p <file_params> -c <file_chains>
```

- __Decimate__ a bmDCA model to 2% density:

```bash
adabmDCA train -m edDCA -d <fasta_file> -p <file_params> -c <file_chains>
```

- __Train and decimate__ a bmDCA model to 2% density:

```bash
adabmDCA train -m edDCA -d <fasta_file>
```

- __Generate sequences__ from a trained model:

```bash
adabmDCA sample -p <file_params> -d <fasta_file> -o <output_folder> --ngen <num_gen>
```

- __Score a sequence set__:

```bash
adabmDCA energies -d <fasta_file> -p <file_params> -o <output_folder>
```

- __Generate a single mutant library__ from a wild type:

```bash
adabmDCA DMS -d <WT> -p <file_params> -o <output_folder>
```

- __Compute contact scores__ via Frobenius norm:

```bash
adabmDCA contacts -p <file_params> -o <output_folder>
```

- __Reintegrate__ DCA model from experiments:

```bash
adabmDCA reintegrate -d <nat_msa> -o <output_folder> --reint <reint_msa> --adj <adj_vector> --alphabet <protein/rna>
```

- __Train/test split__ for homologous sequences:

```bash
adabmDCA profmark -t1 <t1> -t2 <t2> --bestof <n_trials> <output_prefix> <input_msa>
```

