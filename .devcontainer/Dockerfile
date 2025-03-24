FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Python and pip
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3 python3-pip python3-venv \
    # Required for geopandas
    && apt-get -y install --no-install-recommends libgdal-dev

# Set Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]