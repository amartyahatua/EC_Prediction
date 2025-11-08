# EC Classification with Mechanistic Interpretability

Analysis of how protein language models (ESM-2, ProtT5, etc.) encode enzyme function across layers.

## Quick Start

### Step 1: Get Data

You have **3 options**:

#### Option A: Use Ashley's GENSEC Dataset (RECOMMENDED)
```bash
# Ask Ashley for the GENSEC file, then:
python ec_pipeline.py --data /path/to/gensec.csv --type gensec
```

#### Option B: Download Sample Data from UniProt (for testing)
```bash
# Download ~800 proteins from UniProt (takes 2-3 minutes)
python download_sample_data.py

# This creates sample_ec_data.csv, then run:
python ec_pipeline.py --data sample_ec_data.csv --min-samples 10
```

#### Option C: Use Your Own CSV File
```bash
# Your CSV should have columns for sequence and EC number
python ec_pipeline.py \
    --data your_file.csv \
    --seq-col sequence \
    --ec-col EC
```

### Step 2: Run Analysis

Basic usage:
```bash
python ec_pipeline.py --data your_data.csv
```

With options:
```bash
python ec_pipeline.py \
    --data your_data.csv \
    --model esm2_t12_35M_UR50D \
    --min-samples 15 \
    --max-samples 200 \
    --output results.png
```

## Files Included

1. **ec_pipeline.py** - Complete end-to-end analysis pipeline
2. **ec_classification_torchub.py** - Original working example with toy data
3. **load_real_datasets.py** - Data loading utilities for various formats
4. **download_sample_data.py** - Download test data from UniProt

## Requirements

```bash
pip install torch fair-esm scikit-learn matplotlib pandas
```

## Command Line Options

```
--data PATH          Path to your data file (REQUIRED)
--type TYPE          Data type: 'csv' or 'gensec' (default: csv)
--seq-col NAME       Sequence column name (default: sequence)
--ec-col NAME        EC number column name (default: EC)
--model NAME         ESM model to use (default: esm2_t12_35M_UR50D)
--min-samples N      Min samples per EC class (default: 10)
--max-samples N      Max samples per EC class for balancing (optional)
--output PATH        Output plot filename (default: ec_results.png)
```

## Available Models

Small (fast, for testing):
- `esm2_t12_35M_UR50D` - 12 layers, 35M parameters (default)

Medium:
- `esm2_t30_150M_UR50D` - 30 layers, 150M parameters
- `esm2_t33_650M_UR50D` - 33 layers, 650M parameters

Large (slow, best accuracy):
- `esm2_t36_3B_UR50D` - 36 layers, 3B parameters
- `esm2_t48_15B_UR50D` - 48 layers, 15B parameters

## Expected Runtime

With default settings (esm2_t12_35M_UR50D):
- First run: ~5-10 minutes (downloads model)
- Subsequent runs: ~2-5 minutes per 500 sequences

## Output

The script produces:

1. **Console output** showing:
   - Data loading statistics
   - Processing progress
   - Layer-by-layer accuracy
   - Best layer for each EC hierarchy level

2. **Plot (PNG file)** with 4 subplots:
   - EC Level 1 (class): e.g., "3" = Hydrolases
   - EC Level 2 (subclass): e.g., "3.4" = Acts on peptide bonds
   - EC Level 3 (sub-subclass): e.g., "3.4.21" = Serine endopeptidases
   - EC Level 4 (specific enzyme): e.g., "3.4.21.1" = Chymotrypsin

3. **Summary table**:
   ```
   BEST LAYERS FOR EACH EC LEVEL:
   ======================================================================
   EC Level 1: Layer  3 (Val Acc = 0.892)
   EC Level 2: Layer  7 (Val Acc = 0.756)
   EC Level 3: Layer  9 (Val Acc = 0.634)
   EC Level 4: Layer 11 (Val Acc = 0.521)
   ======================================================================
   ```

## Interpretation

**Key findings to look for:**

1. **Hierarchical processing**: Do coarse features (Level 1) emerge in early layers while fine-grained features (Level 4) emerge in late layers?

2. **Cross-model consistency**: Run with multiple models - do they show similar patterns?

3. **Functional organization**: Which layers encode the most functional information?

## Example Analyses

### For your meeting with Ashley:

```bash
# 1. Quick test with sample data
python download_sample_data.py
python ec_pipeline.py --data sample_ec_data.csv --output test_results.png

# 2. Full analysis with GENSEC (once you have it)
python ec_pipeline.py \
    --data gensec.csv \
    --type gensec \
    --min-samples 20 \
    --max-samples 150 \
    --output gensec_results.png

# 3. Compare multiple models
python ec_pipeline.py --data gensec.csv --model esm2_t12_35M_UR50D --output results_12layer.png
python ec_pipeline.py --data gensec.csv --model esm2_t33_650M_UR50D --output results_33layer.png
```

### For your paper:

```bash
# Comprehensive analysis across models
for model in esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D
do
    python ec_pipeline.py \
        --data your_dataset.csv \
        --model $model \
        --output results_${model}.png
done
```

## Data Format Requirements

Your CSV file should have:
- **Sequence column**: Protein sequences (amino acid strings)
- **EC column**: EC numbers in format "X.X.X.X" (e.g., "3.4.21.1")

Example CSV:
```csv
sequence,EC
MKTAYIAKQRQISFVK...,3.4.21.1
MKWVTFISLLFLFSSA...,3.4.21.5
MQIFVKTLTGKTITLE...,1.1.1.1
```

## Troubleshooting

**"No such file or directory"**
- Make sure your data file path is correct
- Use absolute paths if needed: `/full/path/to/data.csv`

**"Column not found"**
- Check your column names with: `head -1 your_data.csv`
- Specify correct names with `--seq-col` and `--ec-col`

**"Not enough samples"**
- Lower `--min-samples` (default is 10)
- Or get more data for rare EC classes

**Model download fails**
- Check internet connection
- Models download to `~/.cache/torch/hub/`
- First download takes 5-10 minutes

**Out of memory**
- Use smaller model: `--model esm2_t12_35M_UR50D`
- Process fewer samples: `--max-samples 100`
- Use CPU if needed (automatically detected)

## Next Steps for Paper

1. **Validate with real data** (GENSEC from Ashley)
2. **Compare multiple models** (ESM-2, ProtT5, EVO)
3. **Analyze attention patterns** (which residues do layers attend to?)
4. **Activation patching** (test causal importance of layers)
5. **Relate to structure** (do layers learn active sites?)

## Questions?

Contact Amartya or check:
- ESM models: https://github.com/facebookresearch/esm
- UniProt API: https://www.uniprot.org/help/api
- VenusFactory: https://github.com/ai4protein/VenusFactory
