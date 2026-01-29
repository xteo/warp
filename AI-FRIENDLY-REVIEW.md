# AI-Friendly Repository Review: NVIDIA Warp

**Review Date:** January 2026
**Repository:** NVIDIA Warp - High-Performance Python Simulation Framework
**Purpose:** Comprehensive assessment of AI-friendliness and recommendations for improvement

---

## Executive Summary

The Warp repository has **solid foundational AI support** through its `AGENTS.md` file and CodeRabbit integration, but **lacks multi-platform AI assistant configurations** and has **documentation gaps** that hinder AI assistants from fully understanding domain-specific concepts. This review identifies missing structures, best practices, and opportunities to build "Warp Skills" that would help developers leverage AI effectively when working with Warp.

### Overall AI-Friendliness Score: 6.5/10

| Category | Score | Notes |
|----------|-------|-------|
| AI Configuration Files | 5/10 | Only AGENTS.md and CodeRabbit; no Cursor/Copilot configs |
| Documentation Quality | 7/10 | Good API docs, weak on "why" explanations |
| Code Examples | 6/10 | Comprehensive but lack narrative explanation |
| Progressive Learning | 4/10 | No structured curriculum or difficulty markers |
| Cross-Platform Support | 4/10 | Only Claude-specific; no Cursor, Copilot, or MCP support |

---

## Part 1: Current State Assessment

### 1.1 Existing AI Configurations

#### What Exists

| File | Purpose | Quality |
|------|---------|---------|
| `AGENTS.md` | Comprehensive AI agent instructions | **Good** - 150+ lines covering architecture, commands, style |
| `CLAUDE.md` | Claude Code integration | **Minimal** - Just references AGENTS.md |
| `.coderabbit.yml` | Automated code review | **Good** - Configured with path filters |

#### Content Analysis of AGENTS.md

**Strengths:**
- Clear project overview and architecture explanation
- Explicit Python execution guidelines (`uv run`)
- Testing requirements clearly stated (unittest, not pytest)
- Code style and docstring conventions documented
- CI/CD dual-pipeline awareness (GitHub + GitLab)

**Weaknesses:**
- No boundary definitions (what agents should never do)
- No tiered permission system (always/ask/never)
- No code examples showing patterns
- No references to specific file locations for common tasks
- Missing domain-specific Warp concepts explanation

### 1.2 Missing AI Configurations

| Configuration | Status | Impact |
|---------------|--------|--------|
| `.cursorrules` / `.cursor/rules/` | **Missing** | Cursor IDE users have no project-specific guidance |
| `.github/copilot-instructions.md` | **Missing** | GitHub Copilot users lack context |
| MCP Server configurations | **Missing** | No Model Context Protocol support |
| Specialized domain rules | **Missing** | No FEM/physics/GPU-specific AI guidance |
| `.ai/` or `prompts/` directory | **Missing** | No reusable AI prompts for common tasks |

### 1.3 Documentation Assessment

#### API Documentation: **Strong**
- Auto-generated from docstrings via Sphinx
- Type stubs available (`__init__.pyi` - 222KB)
- Cross-references between modules

#### Conceptual Documentation: **Moderate**
- `docs/deep_dive/` covers advanced topics
- `docs/user_guide/` provides tutorials
- Missing: "mental model" documentation for AI assistants

#### Example Documentation: **Weak**
- 72 examples across 8 categories
- Consistent 3-line headers but sparse inline comments
- No difficulty markers or learning paths
- Domain knowledge assumed (physics, PDEs, GPU architecture)

---

## Part 2: Best Practices Gap Analysis

### 2.1 AGENTS.md/CLAUDE.md Best Practices

Based on research from GitHub (2,500+ repo analysis), HumanLayer, and the AGENTS.md standard:

| Best Practice | Warp Status | Recommendation |
|---------------|-------------|----------------|
| Under 300 lines | ✅ ~100 lines | Good |
| Commands with flags | ✅ Present | Good |
| Tech stack versions | ⚠️ Partial | Add Python/CUDA versions |
| Code examples | ❌ Missing | Add syntax examples |
| Three-tier boundaries | ❌ Missing | Add always/ask/never sections |
| File:line references | ❌ Missing | Add pointers to key files |
| Domain glossary | ❌ Missing | Add Warp-specific terms |

### 2.2 Recommended AGENTS.md Structure

```markdown
## Boundaries

### Always Do
- Run tests before committing changes to warp/_src/
- Use `wp.` namespace for public API, `warp._src.` for internals
- Follow Google-style docstrings

### Ask First
- Changes to native/ C++/CUDA code (requires rebuild)
- New dependencies in pyproject.toml
- Modifications to builtins.py (affects code generation)

### Never Do
- Run `wp.clear_kernel_cache()` outside `if __name__ == "__main__":`
- Use pytest (project uses unittest)
- Modify auto-generated files (__init__.pyi)
```

### 2.3 Missing Cursor Rules

Warp should have `.cursor/rules/` with specialized rules:

| Rule File | Purpose |
|-----------|---------|
| `warp-kernels.mdc` | @wp.kernel, @wp.func patterns |
| `warp-types.mdc` | Array types, structs, generics |
| `warp-testing.mdc` | unittest patterns, test discovery |
| `warp-native.mdc` | C++/CUDA code guidelines |
| `warp-fem.mdc` | Finite element domain rules |

### 2.4 GitHub Copilot Configuration

Should add `.github/copilot-instructions.md`:

```markdown
# Copilot Instructions for NVIDIA Warp

## Context
Warp is a Python framework that JIT-compiles to GPU/CPU kernels.
Key decorators: @wp.kernel, @wp.func, @wp.struct

## Code Patterns
- Kernels use wp.tid() for thread index, cannot return values
- Launch kernels with wp.launch(kernel, dim=N, inputs=[...])
- Arrays wrap device memory: wp.array(data, dtype=wp.float32)

## Testing
- Use unittest, NOT pytest
- Run: uv run warp/tests/test_<module>.py
```

---

## Part 3: Skills/Knowledge Resources to Build

### 3.1 Proposed "Warp AI Skills" Architecture

A comprehensive Warp AI Skills package should include:

```
warp-ai-skills/
├── cursorrules/
│   ├── warp-core.mdc           # Core API patterns
│   ├── warp-kernels.mdc        # Kernel development
│   ├── warp-fem.mdc            # FEM module
│   ├── warp-optimization.mdc   # Differentiable programming
│   └── warp-interop.mdc        # PyTorch/JAX integration
├── copilot/
│   └── instructions.md         # GitHub Copilot instructions
├── prompts/
│   ├── debug-kernel.md         # Debugging kernel issues
│   ├── optimize-memory.md      # Memory optimization
│   ├── port-numpy-to-warp.md   # Migration guide
│   └── benchmark-comparison.md # Performance analysis
├── examples/
│   └── annotated/              # Heavily commented examples
└── AGENTS.md                   # Enhanced AGENTS.md
```

### 3.2 Core Knowledge Areas for Skills

| Skill Area | What It Should Teach | Priority |
|------------|---------------------|----------|
| **Kernel Basics** | @wp.kernel syntax, wp.tid(), wp.launch() | High |
| **Type System** | wp.array, dtype, device management | High |
| **Differentiation** | wp.Tape, gradient computation, adjoint kernels | High |
| **Memory Patterns** | Array allocation, device transfers, views | Medium |
| **Interoperability** | NumPy, PyTorch, JAX integration | Medium |
| **FEM Domain** | Finite element concepts, variational methods | Medium |
| **GPU Optimization** | Tiles, memory coalescing, occupancy | Advanced |
| **Native Extension** | C++/CUDA custom kernels | Advanced |

### 3.3 Specific Prompts to Develop

#### Prompt 1: "Port NumPy Code to Warp"
```markdown
You are helping port NumPy array operations to Warp GPU kernels.

Key transformations:
- np.array → wp.array with explicit dtype
- Vectorized ops → @wp.kernel with explicit loops
- Broadcasting → Manual indexing with wp.tid()

Example:
# NumPy
result = a + b * c

# Warp
@wp.kernel
def compute(a: wp.array(dtype=float), b: wp.array(dtype=float),
            c: wp.array(dtype=float), result: wp.array(dtype=float)):
    i = wp.tid()
    result[i] = a[i] + b[i] * c[i]
```

#### Prompt 2: "Debug Kernel Compilation Error"
```markdown
When encountering Warp kernel compilation errors:

1. Check decorator usage (@wp.kernel for entry points, @wp.func for helpers)
2. Verify all types are Warp-compatible (no Python objects in kernels)
3. Ensure array dimensions match launch dimensions
4. Check for unsupported Python constructs (list comprehensions, etc.)

Common errors:
- "unsupported type" → Use wp.float32, wp.int32, etc.
- "cannot return" → Kernels write to output arrays, not return
- "undefined function" → Missing @wp.func decorator
```

#### Prompt 3: "Optimize Warp Kernel Performance"
```markdown
Performance optimization checklist for Warp kernels:

Memory:
- [ ] Use contiguous array access patterns
- [ ] Minimize host-device transfers
- [ ] Reuse arrays instead of reallocating

Compute:
- [ ] Use wp.tile operations for matrix math
- [ ] Consider graph capture for repeated launches
- [ ] Profile with wp.ScopedTimer

Parallelism:
- [ ] Match launch dimensions to problem size
- [ ] Avoid thread divergence in conditionals
- [ ] Use atomic operations sparingly
```

### 3.4 Annotated Example Structure

Each skill should include annotated examples like:

```python
"""
Example: Particle System with Collision Detection

SKILL LEVEL: Intermediate
PREREQUISITES: Kernel basics, array types
CONCEPTS: Spatial hashing, neighbor queries, physics integration

This example demonstrates:
1. How to use wp.HashGrid for O(1) neighbor lookups
2. Particle physics integration patterns
3. Collision response calculation
"""

import warp as wp

# WHY: HashGrid provides O(1) spatial queries vs O(n²) brute force
# WHEN: Use when particles interact within a fixed radius
grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128)

@wp.kernel
def integrate_particles(
    positions: wp.array(dtype=wp.vec3),  # Current particle positions
    velocities: wp.array(dtype=wp.vec3), # Current velocities
    forces: wp.array(dtype=wp.vec3),     # Accumulated forces
    dt: float,                            # Timestep (typically 1/60 for real-time)
):
    """
    Semi-implicit Euler integration for particle physics.

    WHY semi-implicit: More stable than explicit Euler for oscillatory systems
    ALTERNATIVE: Verlet integration for energy conservation
    """
    tid = wp.tid()

    # Update velocity first (semi-implicit)
    # WHY: Reduces energy gain in spring systems
    velocities[tid] = velocities[tid] + forces[tid] * dt

    # Then update position with new velocity
    positions[tid] = positions[tid] + velocities[tid] * dt
```

---

## Part 4: Existing Resources Inventory

### 4.1 What Already Exists for Warp

| Resource | Location | Usefulness for AI |
|----------|----------|-------------------|
| AGENTS.md | Repo root | Good foundation |
| API Docs | nvidia.github.io/warp | Comprehensive reference |
| 72 Examples | warp/examples/ | Code patterns, weak narrative |
| 7 Notebooks | notebooks/ | Progressive learning |
| GTC Sessions | NVIDIA On-Demand | Video explanations |
| Publications | PUBLICATIONS.md | Academic context |

### 4.2 External Resources That Could Be Leveraged

| Resource | URL | Applicable Patterns |
|----------|-----|---------------------|
| awesome-cursorrules CUDA rules | github.com/PatrickJS/awesome-cursorrules | GPU memory patterns |
| PyTorch Cursor Rules | cursor.directory/rules/pytorch | Tensor operations |
| CUDA Best Practices | cursorrules.org | Optimization guidance |
| NeMo Cursor Rules Guide | NVIDIA docs | Rule file structure |

### 4.3 No Existing Warp-Specific AI Configs Found

Web searches confirmed:
- No Warp entries on cursor.directory
- No community .cursorrules for Warp
- No GitHub Copilot instructions for Warp
- No MCP servers for Warp

**Opportunity:** Creating Warp AI Skills would be novel and fill a gap.

---

## Part 5: Recommendations

### 5.1 Immediate Actions (High Priority)

1. **Enhance AGENTS.md with boundaries**
   ```markdown
   ## Boundaries

   ### Always
   - Use `uv run` for Python execution
   - Run tests after modifying warp/_src/

   ### Ask First
   - Changes to builtins.py (triggers stub regeneration)
   - Adding native C++/CUDA code

   ### Never
   - Use pytest (project uses unittest)
   - Call wp.clear_kernel_cache() in test files
   ```

2. **Add code examples to AGENTS.md**
   - Show @wp.kernel pattern
   - Show wp.launch() usage
   - Show array creation patterns

3. **Create `.github/copilot-instructions.md`**
   - Copy key sections from AGENTS.md
   - Add Warp-specific code patterns

### 5.2 Medium-Term Actions

4. **Create `.cursor/rules/` directory**
   - `warp-core.mdc` - Core API patterns
   - `warp-testing.mdc` - Testing conventions
   - `warp-native.mdc` - C++/CUDA guidelines

5. **Add difficulty markers to examples**
   - Tag each example as Beginner/Intermediate/Advanced
   - Create `examples/README.md` with learning paths

6. **Create annotated example variants**
   - Take 5-10 core examples
   - Add extensive "why" comments
   - Document design decisions

### 5.3 Long-Term Actions

7. **Build Warp AI Skills package**
   - Standalone repository or docs section
   - Cursor rules, Copilot instructions, prompts
   - Publish to cursor.directory

8. **Create domain-specific guides**
   - FEM concepts for AI assistants
   - GPU optimization patterns
   - Physics simulation fundamentals

9. **Consider MCP server**
   - Warp documentation server
   - Kernel analysis tools
   - Example search

---

## Part 6: Proposed File Additions

### 6.1 Enhanced AGENTS.md (Proposed)

```markdown
# AGENTS.md

## Project Overview
Warp is a Python framework for high-performance simulation. It JIT-compiles
Python to GPU/CPU kernels.

## Quick Reference

### Key Patterns
```python
import warp as wp

# Define a kernel (runs in parallel)
@wp.kernel
def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    i = wp.tid()  # Get thread index
    b[i] = a[i] * 2.0

# Launch kernel
wp.launch(my_kernel, dim=1000, inputs=[input_array, output_array])
```

### Commands
- Test single file: `uv run warp/tests/test_<name>.py`
- Build native libs: `uv run build_lib.py`
- Format code: `uvx pre-commit run -a`
- Build docs: `uv run --extra docs build_docs.py`

## Architecture
[existing content...]

## Boundaries

### Always Do
- Use `uv run` for all Python execution
- Follow Google-style docstrings
- Run tests after modifying `warp/_src/`

### Ask First
- Adding/modifying `warp/native/` C++/CUDA code
- Changes to `builtins.py` (requires doc regeneration)
- New dependencies in `pyproject.toml`

### Never Do
- Use pytest (project uses unittest)
- Call `wp.clear_kernel_cache()` in test files
- Modify auto-generated `__init__.pyi`
- Import from `warp._src` in public-facing code

## Key File Locations
- Kernel codegen: `warp/_src/codegen.py:generate_kernel`
- Built-in functions: `warp/_src/builtins.py`
- Runtime context: `warp/_src/context.py:Runtime`
- Native bindings: `warp/native/warp.cpp`
- Test utilities: `warp/tests/unittest_utils.py`

[rest of existing content...]
```

### 6.2 New `.github/copilot-instructions.md`

```markdown
# GitHub Copilot Instructions for NVIDIA Warp

## What is Warp?
Warp is a Python framework that JIT-compiles code to run on CUDA GPUs or CPUs.
It's designed for simulation, graphics, and differentiable programming.

## Key Decorators
- `@wp.kernel` - Parallel code entry point (like CUDA __global__)
- `@wp.func` - Device function callable from kernels (like CUDA __device__)
- `@wp.struct` - Custom data structures for kernels

## Core Patterns

### Kernel Definition
```python
@wp.kernel
def compute(input: wp.array(dtype=wp.float32),
            output: wp.array(dtype=wp.float32)):
    i = wp.tid()  # Thread index
    output[i] = input[i] * 2.0
```

### Kernel Launch
```python
wp.launch(kernel=compute, dim=n, inputs=[input_arr, output_arr], device="cuda")
```

### Array Creation
```python
arr = wp.array(np_array, dtype=wp.float32, device="cuda")
```

## Testing
- Framework: unittest (NOT pytest)
- Run: `uv run warp/tests/test_<module>.py`
- New tests go in `warp/tests/`, add to `unittest_suites.py`

## Code Style
- Docstrings: Google-style
- Formatting: Run `uvx pre-commit run -a`
- Imports: Use `warp` for public API, `warp._src` for internals only
```

### 6.3 New `.cursor/rules/warp-core.mdc`

```markdown
---
description: Core NVIDIA Warp patterns for kernel development
globs: ["**/*.py"]
alwaysApply: false
---

# Warp Core Patterns

## Kernel Development

### Kernel Structure
- Use `@wp.kernel` decorator for parallel entry points
- Use `@wp.func` for helper functions callable from kernels
- Get thread index with `wp.tid()`
- Kernels cannot return values; write to output arrays

### Type Safety
- Always specify dtype: `wp.array(dtype=wp.float32)`
- Use Warp types in kernels: `wp.float32`, `wp.vec3`, `wp.mat33`
- Python objects cannot be used inside kernels

### Common Pitfalls
- Don't use Python list/dict inside kernels
- Don't use Python range() - use wp.range()
- Don't return from kernels - write to output arrays
- Don't call wp.launch() inside kernels

## Testing
- Use unittest, not pytest
- Run single test: `uv run warp/tests/test_<name>.py`
- Add new test files to `unittest_suites.py`

## Memory Management
- Prefer reusing arrays over reallocating
- Use `wp.synchronize()` before reading results on CPU
- Use `array.numpy()` to get NumPy view (implicit sync)
```

---

## Appendix A: Research Sources

### AI-Friendly Repository Best Practices
- GitHub Blog: "How to write a great agents.md" (2,500+ repo analysis)
- HumanLayer: "Writing a good CLAUDE.md"
- AGENTS.md official specification (agents.md)
- Dometrain: "Creating the Perfect CLAUDE.md"

### Cursor Rules
- awesome-cursorrules repository (37.5k stars)
- cursorrules.org CUDA best practices
- NVIDIA NeMo Cursor Rules Developer Guide

### GPU Programming
- NVIDIA Warp Documentation
- NVIDIA Developer Blog on Warp
- GTC 2024 Warp session

---

## Appendix B: Competitive Analysis

| Framework | AI Config Quality | Notes |
|-----------|------------------|-------|
| PyTorch | Good | Has cursor.directory entry, community rules |
| JAX | Moderate | Some community resources |
| Taichi | Poor | Minimal AI guidance |
| **Warp** | **Moderate** | Good AGENTS.md, missing multi-platform |
| CUDA (general) | Good | Multiple cursor rules available |

---

## Conclusion

The Warp repository has a solid foundation for AI-friendliness but significant opportunities remain:

1. **Multi-platform support** - Add Cursor and Copilot configurations
2. **Enhanced documentation** - Add "why" explanations and learning paths
3. **Community resources** - Publish Warp Skills to cursor.directory
4. **Domain guidance** - Create AI-digestible explanations of FEM/physics concepts

Implementing these recommendations would make Warp one of the most AI-friendly scientific computing frameworks available.
