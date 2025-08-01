FROM continuumio/miniconda3

WORKDIR /main

COPY env.clean.yml .
COPY . .

RUN conda env create -f env.clean.yml
RUN conda clean -afy

# Set shell for subsequent RUNs, optional
SHELL ["conda", "run", "-n", "real-estate", "/bin/bash", "-c"]

# Use conda run explicitly in CMD
CMD ["conda", "run", "--no-capture-output", "-n", "real-estate", "python", "main.py"]