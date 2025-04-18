# Lab: Multi-Architecture Docker Images

## Objectives
- Understand why multi-architecture Docker images are important
- Learn different approaches to building multi-architecture Docker images
- Modify the existing flowers classification project to support multiple architectures
- Test the multi-architecture setup

## Introduction

The current project uses Docker and Docker Compose to package and run several services:
- A Python-based flower classification application
- MLflow for experiment tracking
- Prometheus for monitoring
- Grafana for visualization

Currently, these Docker images are built for a single architecture (the architecture of the machine they're built on, likely amd64/x86_64). This limits where the containers can run. In this lab, we'll explore how to modify our approach to support multiple CPU architectures, enabling our application to run on different platforms.

## Why Multi-Architecture Matters

### The Problem

Docker images are architecture-specific by default. When you build a Docker image, it's built for the CPU architecture of the machine you're building on. For example:

- Images built on an Intel/AMD machine (x86_64/amd64) won't run on ARM-based systems like Apple M1/M2 (arm64)
- Images built on Apple Silicon (arm64) won't run on traditional x86_64 servers

This creates compatibility issues in different environments:

1. **Development vs. Production**: If developers use different hardware than production servers
2. **Cloud Deployment**: When deploying to cloud providers with different architectures
3. **Edge Computing**: When deploying to IoT or edge devices that often use ARM processors
4. **Team Collaboration**: When team members use different machine architectures

### Benefits of Multi-Architecture Images

1. **Portability**: Run the same images across different types of hardware
2. **Consistent Environment**: Ensure identical behavior regardless of underlying architecture
3. **Simplified CI/CD**: Maintain a single image tag/reference for multiple architectures

---

> **Summary of Key Correction:**
> - `docker-compose build` does NOT support multi-architecture builds. Use `docker buildx build` with the `--platform` flag and `--push`.
> - Reference built images in your compose file, do not rely on local builds for multi-arch support.
4. **Future-Proofing**: Support for emerging hardware platforms

## Prerequisites

Before starting, ensure you have:

- Docker 19.03+ with Buildx enabled (Docker Desktop or recent Docker Engine)
- QEMU emulation binaries installed (for cross-building)
- Access to a Docker registry (e.g., Docker Hub)
- (Optional) Docker Compose v2 for advanced features

## Multi-Architecture Approaches in Docker

Docker supports several approaches to create multi-architecture images. The recommended method is using Buildx.

### 1. Docker Buildx with Multi-Platform Builds

[Docker Buildx](https://docs.docker.com/buildx/working-with-buildx/) is the recommended modern tool for building multi-architecture images. It's built into Docker Desktop and recent Docker CLI versions.

#### Step-by-Step: Building and Pushing Multi-Arch Images

1. **Enable Buildx (if not already):**
   ```bash
   docker buildx create --use
   ```
2. **Set up QEMU emulation (for cross-building):**
   ```bash
   docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
   ```
3. **Build and push a multi-arch image:**
   ```bash
   docker buildx build --platform linux/amd64,linux/arm64 -t <your-username>/<image-name>:<tag> --push .
   ```
   Replace `<your-username>`, `<image-name>`, and `<tag>` as appropriate.

4. **Repeat for each service** (if you have multiple Dockerfiles/services).

#### Referencing Multi-Arch Images in docker-compose.yml

Once your images are built and pushed, update your `docker-compose.yml` to use the pushed images:

```yaml
services:
  flower-app:
    image: <your-username>/<image-name>:<tag>
  mlflow:
    image: <your-username>/<mlflow-image>:<tag>
  # ... other services
```

All collaborators and deployment environments should pull these images, not build locally.

#### (Optional) Advanced: Multi-Service Builds with Buildx Bake

For advanced users, [Buildx Bake](https://docs.docker.com/build/bake/) can build multiple services (like Compose) with multi-arch support. This is optional and not required for most student projects.

#### Testing Multi-Arch Images

- **On Local Hardware:** Use real hardware (e.g., x86_64 and arm64 machines) if available.
- **Using Emulation:**
  ```bash
  docker run --platform=linux/arm64 <your-username>/<image-name>:<tag>
  docker run --platform=linux/amd64 <your-username>/<image-name>:<tag>
  ```
- **On Cloud:** Deploy to cloud VMs of different architectures to confirm compatibility.

---

#### Check if buildx is available
docker buildx version

#### Create a new builder instance with multi-architecture support
```bash
docker buildx create --name multiarch-builder --use

# Verify the builder and supported platforms

docker buildx inspect

# Building Multi-Architecture Images

docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/appname:tag --push .
```

### 2. Docker Manifest Lists

This approach involves building separate images for each architecture and then combining them into a manifest list.

```bash
# Build for different architectures
docker build -t yourusername/appname:amd64 --platform linux/amd64 .
docker build -t yourusername/appname:arm64 --platform linux/arm64 .

# Create and push a manifest list
docker manifest create yourusername/appname:latest \
  yourusername/appname:amd64 \
  yourusername/appname:arm64

docker manifest push yourusername/appname:latest
```

### 3. QEMU-Based Cross-Compilation

This approach uses QEMU to emulate different architectures during the build process.

```bash
# Set up QEMU for cross-architecture builds
docker run --privileged --rm tonistiigi/binfmt --install all

# Now you can build for other architectures
docker build --platform linux/arm64 -t yourusername/appname:arm64 .
```

## Modifying Our Flower Classification Project

Let's update our project to support multi-architecture builds. Here are the changes needed:

### 1. Update Dockerfile and Dockerfile.mlflow

Our current Dockerfiles use `python:3.10-slim` as the base image. This image supports multiple architectures, but we need to ensure our build process handles them correctly.

Both Dockerfiles can remain largely unchanged as the base image already supports multiple architectures. However, we should be aware of the following:

- PyTorch installation in the main Dockerfile may need architecture-specific handling
- Any architecture-specific optimizations or libraries should be conditionally installed

### 2. Update docker-compose.yml

We'll add build configuration to support multi-architecture:

```yaml
version: '3'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      platforms:
        - linux/amd64
        - linux/arm64
    # Rest of configuration remains the same

  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
      platforms:
        - linux/amd64
        - linux/arm64
    # Rest of configuration remains the same

  # Prometheus and Grafana services remain unchanged as they use official
  # images that already support multiple architectures
```

### 3. Create a Build Script

Create a `build-multiarch.sh` script in the project root:

```bash
#!/bin/bash
# Script to build multi-architecture Docker images for our project

# Set up buildx if not already configured
docker buildx inspect multiarch-builder >/dev/null 2>&1 || \
  docker buildx create --name multiarch-builder --use

# Build and push the app image
echo "Building multi-architecture app image..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t masorian06/flowers-classification:latest \
  --push \
  -f Dockerfile .

# Build and push the MLflow image
echo "Building multi-architecture MLflow image..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t masorian06/flowers-mlflow:latest \
  --push \
  -f Dockerfile.mlflow .

echo "Multi-architecture builds complete!"
```

Make the script executable:
```bash
chmod +x build-multiarch.sh
```

### 4. Handle PyTorch Installation

Since PyTorch can be architecture-specific, modify the main Dockerfile to handle different architectures:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch based on architecture
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      pip install torch torchvision torchaudio; \
    else \
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Rest of the Dockerfile remains the same
COPY . .
RUN mkdir -p models predictions data/raw data/processed data/external
ENV PYTHONPATH=/app
ENV FLASK_APP=api_main.py
EXPOSE 9000
CMD ["python", "api_main.py"]
```

## Testing the Multi-Architecture Setup

After implementing these changes, you should test your multi-architecture images:

1. Build the images using the build script:
   ```bash
   ./build-multiarch.sh
   ```

2. Verify that the images support multiple architectures:
   ```bash
   docker buildx imagetools inspect yourusername/flowers-classification:latest
   ```

3. Test the images on different architectures:
   - On an amd64 machine: `docker-compose up`
   - On an arm64 machine: `docker-compose up`

4. For local testing on a different architecture, you can use:
   ```bash
   docker run --platform linux/arm64 yourusername/flowers-classification:latest
   ```

### Understanding How Multi-Architecture Images Work

Let's clear up a common misconception: Docker multi-architecture images aren't single "fat" images containing code for all architectures bundled together. Instead, think of them like a smart router:

1. When you create a multi-architecture image (whether using Buildx or manifest commands), you're actually creating:
   - Separate images for each architecture (e.g., one for amd64, one for arm64)
   - A manifest list (or "image index") that acts like a table of contents

2. When someone runs `docker pull yourusername/flowers-classification:latest`:
   - Docker checks what CPU architecture their machine has
   - It looks at the manifest list to find the matching image
   - It downloads only the appropriate image for that machine

3. This happens automatically - users don't need to specify which architecture they want!

The beauty of this system is that two people with different computer types (like an Intel laptop and an M1 Mac) can use the exact same Docker command, but each will get the version that works for their specific machine.

## Common Issues and Solutions

1. **Build Errors for Specific Architectures**:
   - Check that all dependencies support your target architectures
   - Use conditional installation based on architecture

2. **Performance Issues on Emulated Architectures**:
   - Native builds are preferred for performance-critical components
   - Consider using architecture-specific optimizations where needed

3. **Increased Build Time**:
   - Multi-architecture builds take longer, especially with emulation
   - Consider using a CI/CD pipeline with architecture-specific builders

4. **Image Size Concerns**:
   - Multi-architecture images can be larger when using manifests
   - Use proper tagging and versioning strategies

## Conclusion

By implementing multi-architecture support in our Docker project, we've improved the portability and compatibility of our flower classification application. This ensures our containerized services can run consistently across different environments and hardware platforms.

The approaches outlined here, particularly using Docker Buildx, represent current best practices for multi-architecture Docker images. As container technologies evolve, keep an eye on Docker documentation for the latest recommendations.

## Next Steps

I encourage you to experiment with different multi-architecture strategies (e.g., using Docker manifest lists or QEMU-based cross-compilation) and continue learning about the latest best practices in containerization and multi-architecture support.

## Resources for Further Learning

- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [Multi-platform Images with Docker](https://www.docker.com/blog/multi-platform-docker-builds/)
- [Docker Manifest Command](https://docs.docker.com/engine/reference/commandline/manifest/)
- [QEMU for Docker Cross-Compilation](https://github.com/tonistiigi/binfmt)
