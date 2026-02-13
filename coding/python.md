Prompt #1
Title: Intelligent File Organizer
Category: Automation
Difficulty: Beginner
Description: Create a script that automatically organizes files in a directory by type, date, and size with intelligent duplicate detection.
The Prompt:
Act as a Python automation expert. Write a file organization script that scans a target directory, categorizes files by extension into subfolders (Documents, Images, Videos, Audio, Archives, Code), handles naming conflicts by appending timestamps, detects duplicate files using MD5 hashing, logs all operations to a CSV file, and includes a dry-run mode for testing. The script must use pathlib for cross-platform compatibility, include progress bars with tqdm, and provide a command-line interface with argparse. Structure the output as: (1) imports and constants, (2) helper functions with docstrings, (3) main organization class, (4) CLI entry point, (5) example usage commands.

Prompt #2
Title: Dynamic Web Scraper with Anti-Detection
Category: Web Scraping
Difficulty: Intermediate
Description: Build a robust scraper that extracts structured data from JavaScript-rendered websites while avoiding detection mechanisms.
The Prompt:
Assume the role of a senior web scraping engineer. Develop a Python scraper using Playwright and BeautifulSoup that extracts product data from an e-commerce site with the following requirements: rotate User-Agents and proxies from a pool, implement random delays between requests (1-5 seconds), handle CAPTCHA detection with notification alerts, extract product name, price, availability, and rating, store data in SQLite with timestamp, retry failed requests with exponential backoff, and export results to both CSV and JSON. Include a configuration file for target URLs and selectors, error logging with rotation, and a rate limiter compliant with robots.txt parsing. Output must include class diagram, main scraper module, database schema, and example config.yaml.


Prompt #3
Title: Real-Time Stock Market Analyzer
Category: Finance Tools
Difficulty: Advanced
Description: Build a streaming data pipeline that processes live stock market data, calculates technical indicators, and generates trading signals.
The Prompt:
As a quantitative Python developer, create a real-time stock analysis system using asyncio and WebSocket connections to stream data from Alpaca or Finnhub API. Implement the following components: (1) asynchronous data ingestion with connection recovery, (2) circular buffer for last 1000 price points, (3) real-time calculation of RSI, MACD, Bollinger Bands, and VWAP using pandas-ta, (4) signal generation engine with configurable thresholds, (5) alert dispatcher via webhook/email for buy/sell signals, (6) SQLite storage for historical signals with performance tracking. Include risk management module with position sizing based on volatility, backtesting framework using historical data, and dashboard with Plotly Dash for visualization. Structure as modular microservices with clear API contracts.


Prompt #4
Title: Secure Password Manager CLI
Category: Cybersecurity
Difficulty: Intermediate
Description: Create a command-line password manager with military-grade encryption and secure credential storage.
The Prompt:
Act as a Python security specialist. Design a password manager CLI using cryptography.fernet and argon2 with these specifications: master password derivation using Argon2id with 64MB memory and 3 iterations, AES-256-GCM encryption for stored credentials, secure random password generation with configurable entropy, breach detection via Have I Been Pwned API integration, clipboard integration with auto-clear after 30 seconds, database encryption with SQLCipher, TOTP generation for 2FA backup codes, and export functionality to encrypted JSON. Include threat model documentation, secure memory handling with explicit zeroing, audit logging for all access attempts, and comprehensive unit tests with 90%+ coverage. Output must include architecture diagram, main modules, and security considerations.


Prompt #5
Title: Async REST API Framework
Category: APIs
Difficulty: Advanced
Description: Build a production-ready asynchronous API framework with automatic documentation and testing.
The Prompt:
As a backend architecture expert, create a Python async API framework using FastAPI, Pydantic v2, and SQLAlchemy 2.0 with these features: automatic OpenAPI 3.1 generation with examples, request/response validation with custom validators, JWT authentication with refresh token rotation, rate limiting with Redis sliding window, background task queue with Celery, database migrations with Alembic, automated testing with pytest-asyncio and testcontainers, metrics collection with Prometheus, structured logging with structlog, and deployment configuration for Docker and Kubernetes. Include generic repository pattern, unit of work implementation, CQRS command handlers, and event sourcing for audit trails. Structure as a cookiecutter template with clear separation of concerns.



Prompt #6
Title: Neural Network Visualizer
Category: Machine Learning
Difficulty: Expert
Description: Create an interactive tool that visualizes neural network architectures, forward passes, and gradient flows in real-time.
The Prompt:
Assume the role of a ML infrastructure engineer. Build a PyTorch neural network visualizer using PyTorchViz, Graphviz, and Streamlit with: dynamic architecture rendering from model definitions, forward pass animation showing tensor transformations, gradient flow heatmaps identifying vanishing/exploding gradients, layer-wise activation statistics with histograms, FLOPs and parameter counting with torchinfo, comparison mode for multiple model variants, and export to interactive HTML. Support CNN, RNN, Transformer, and custom architectures. Include profiling integration with PyTorch Profiler, memory tracking with nvidia-ml-py, and educational tooltips explaining each component. Output must be a pip-installable package with CLI entry point and comprehensive documentation.


Prompt #7
Title: Distributed Task Queue System
Category: System Design
Difficulty: Expert
Description: Design and implement a distributed task processing system with guaranteed delivery and fault tolerance.
The Prompt:
As a distributed systems architect, create a Python task queue system inspired by Celery but lightweight using asyncio, Redis Streams, and gRPC. Implement: at-least-once delivery with idempotency keys, exactly-once processing for critical tasks, priority queues with weighted fair queuing, dead letter queues with exponential backoff retry, worker auto-scaling based on queue depth, circuit breaker pattern for failing services, distributed tracing with OpenTelemetry, and cluster state management with Raft consensus. Include protobuf service definitions, Kubernetes operator for deployment, comprehensive load testing with Locust, and chaos engineering tests with random worker failures. Structure as a monorepo with client SDK, worker runtime, and control plane.


Prompt #8
Title: Interactive Data Dashboard Builder
Category: Data Analysis
Difficulty: Intermediate
Description: Create a drag-and-drop dashboard builder for CSV/Excel files with automatic chart suggestions.
The Prompt:
Act as a Python data visualization expert. Build a Streamlit-based dashboard builder that: accepts CSV/Excel uploads with automatic type inference, suggests optimal chart types using heuristics (cardinality, distribution, correlation), provides drag-and-drop interface with streamlit-draggable, implements pivot table generation with aggregation functions, adds filtering with multi-select and date ranges, supports real-time updates from database connections, exports dashboards to static HTML, and includes collaborative commenting with SQLite backend. Use Plotly for interactive charts, Polars for large dataset processing (>1M rows), and caching with diskcache. Include template gallery with 10 pre-built dashboard layouts and comprehensive user guide.


Prompt #9
Title: Python Code Security Scanner
Category: Cybersecurity
Difficulty: Advanced
Description: Develop a static analysis tool that detects security vulnerabilities, anti-patterns, and compliance violations in Python codebases.
The Prompt:
As a application security engineer, create a Python security scanner using libcst, bandit, and semgrep with: AST-based analysis for SQL injection, XSS, and hardcoded secrets, data flow analysis for taint tracking, SAST rule engine with YAML-defined patterns, SBOM generation with cyclonedx-bom, license compliance checking with FOSSology integration, CI/CD integration with SARIF output, false positive suppression with justification comments, and remediation suggestions with code patches. Include 50+ built-in security rules, custom rule DSL, differential scanning for pull requests, and executive summary generation with risk scoring. Structure as a pre-commit hook, GitHub Action, and standalone CLI with JSON/SARIF/HTML output formats.


Prompt #10
Title: 2D Physics Game Engine
Category: Game Development
Difficulty: Expert
Description: Build a complete 2D physics engine with collision detection, rigid body dynamics, and particle systems.
The Prompt:
Assume the role of a game engine developer. Create a pure Python 2D physics engine using numpy and numba for performance with: spatial hashing for broad-phase collision detection, SAT (Separating Axis Theorem) for narrow-phase, impulse-based rigid body dynamics with restitution and friction, constraint solver for joints and springs, Verlet integration for particle systems, quadtree for efficient spatial queries, and debug visualization with pygame. Include support for polygons, circles, and compound shapes, continuous collision detection for fast objects, sleeping bodies optimization, and deterministic replay system. Create a demo game (Angry Birds clone) 
showcasing all features with level editor. Structure as Cython-accelerated core with pure Python API.


Prompt #11
Title: Smart Git Repository Analyzer
Category: DevOps Scripting
Difficulty: Intermediate
Description: Build a tool that analyzes Git repositories for code quality trends, contributor patterns, and technical debt.
The Prompt:
Act as a DevOps automation expert. Create a Git repository analyzer using GitPython, PyDriller, and matplotlib that: extracts commit history with change metrics, calculates code churn and hotspot identification, generates contributor network graphs, detects refactoring commits vs feature commits, estimates technical debt using code complexity trends, identifies knowledge silos (single contributors), generates release notes automatically from commit messages, and creates interactive HTML reports with timeline visualization. Include integration with GitHub/GitLab APIs for PR analysis, Slack notifications for threshold breaches, and export to JSON for BI tools. Support analysis of monorepos with path filtering and comparative analysis between branches.


Prompt #12
Title: Natural Language SQL Generator
Category: AI Integration
Difficulty: Advanced
Description: Create a system that converts natural language questions into optimized SQL queries with schema awareness.
The Prompt:
As an AI integration specialist, build a NL-to-SQL system using LangChain, SQLAlchemy, and OpenAI/Anthropic APIs with: automatic schema introspection and embedding generation, few-shot prompting with example query bank, query validation with EXPLAIN execution plan analysis, confidence scoring for generated queries, query explanation in natural language, support for complex joins, aggregations, and window functions, and fallback to human clarification for low confidence. Include schema anonymization for privacy, query result caching with Redis, audit logging for all generated queries, and interactive CLI for testing. Structure as FastAPI service with async generation, comprehensive test suite with 100+ query patterns, and Docker deployment configuration.



Prompt #13
Title: High-Frequency Trading Simulator
Category: Finance Tools
Difficulty: Expert
Description: Build a realistic HFT backtesting engine with market impact modeling and latency simulation.
The Prompt:
Assume the role of a quantitative trading systems developer. Create a Python HFT simulator using NumPy, Numba, and asyncio with: tick-level market data replay with nanosecond precision, order book reconstruction from L2 data, market impact models (Almgren-Chriss, Kissell), latency simulation with jitter and packet loss, exchange matching engine emulation with priority queues, strategy API with event-driven architecture, P&L attribution with transaction cost analysis, and regulatory compliance checks (MiFID II). Include visualization of order flow toxicity, Sharpe ratio optimization with walk-forward analysis, and deployment guide for colocation setups. Structure as C++ extension modules with Python bindings for performance-critical paths.



Prompt #14
Title: Automated Infrastructure Provisioner
Category: DevOps Scripting
Difficulty: Advanced
Description: Create a Python-based infrastructure as code tool that provisions cloud resources across multiple providers.
The Prompt:
As a cloud infrastructure architect, build a multi-cloud provisioner using Pulumi SDK, boto3, and azure-mgmt-resource with: declarative YAML/JSON configuration format, dependency graph resolution with topological sort, state management with S3 backend and locking, drift detection and remediation, cost estimation before provisioning, automatic tagging strategy enforcement, secrets injection from HashiCorp Vault, and rollback capabilities with snapshot restoration. Support AWS, Azure, GCP with unified API, include 50+ resource types (compute, storage, networking, serverless), policy as code with Open Policy Agent, and compliance reporting with CIS benchmarks. Structure as CLI tool with CI/CD integration and comprehensive audit logging.



Prompt #15
Title: Genetic Algorithm Optimizer Framework
Category: Algorithms
Difficulty: Intermediate
Description: Build a flexible genetic algorithm framework for optimization problems with parallel evaluation.
The Prompt:
Act as an algorithms specialist. Create a Python GA framework using DEAP, multiprocessing, and numpy with: pluggable chromosome representations (binary, real-valued, permutation), configurable selection operators (tournament, roulette, rank), crossover operators (single-point, uniform, BLX-alpha), mutation with adaptive rates, niching for multimodal optimization, parallel fitness evaluation with ProcessPoolExecutor, checkpointing for long-running optimizations, and convergence detection with automatic termination. Include visualization of fitness landscapes and convergence curves, CMA-ES hybrid for local refinement, and examples for TSP, function optimization, and neural architecture search. Structure as pip-installable package with comprehensive benchmarks against scipy.optimize.




Prompt #16
Title: Real-Time Log Analysis Pipeline
Category: System Design
Difficulty: Advanced
Description: Build a streaming log processing system with pattern detection, anomaly alerting, and automatic clustering.
The Prompt:
As a data platform engineer, create a Python log analysis pipeline using Faust (Kafka Streams), Elasticsearch, and scikit-learn with: structured log parsing with grok patterns, real-time pattern extraction with drain3 algorithm, anomaly detection with isolation forest on log embeddings, automatic alert generation with PagerDuty integration, log clustering for root cause analysis, trend analysis with time series forecasting, and interactive Kibana dashboards. Include support for 20+ log formats (syslog, JSON, Apache, nginx), backpressure handling with circuit breakers, and ML model versioning with MLflow. Structure as Kubernetes-native application with Helm charts and comprehensive observability.


Prompt #17
Title: Smart Contract Auditing Toolkit
Category: Blockchain
Difficulty: Expert
Description: Create a security toolkit for analyzing Ethereum smart contracts with vulnerability detection and formal verification hints.
The Prompt:
Assume the role of a blockchain security researcher. Build a Python toolkit using web3.py, slither, and mythril with: automatic contract decompilation and CFG generation, vulnerability pattern matching for OWASP Top 10, taint analysis for reentrancy and integer overflow, gas optimization suggestions with exact savings calculation, formal verification property generation as hints, upgradeability pattern detection in proxy contracts, and comprehensive PDF report generation. Include integration with Etherscan for source code fetching, historical exploit database matching, and differential analysis between contract versions. Structure as CLI with CI/CD integration and REST API for automated scanning.


Prompt #18
Title: Cross-Platform GUI Application Builder
Category: GUI Apps
Difficulty: Intermediate
Description: Create a modern GUI framework abstraction that compiles to native apps for Windows, macOS, and Linux.
The Prompt:
Act as a desktop application developer. Build a Python GUI framework using Toga, PyOxidizer, and declarative UI definitions with: JSON/YAML-based UI layout specification, reactive data binding with observable properties, native widget set for each platform, async event handling without blocking UI, automatic form validation with error display, file drag-and-drop support, system tray integration, and auto-updater implementation. Include visual form builder with live preview, 20+ built-in widgets, theme customization with CSS-like styling, and packaging scripts for all platforms. Create example applications: code editor, image viewer, and system monitor. Structure as installable package with comprehensive documentation.


Prompt #19
Title: Network Traffic Analyzer
Category: Networking
Difficulty: Advanced
Description: Build a real-time network packet analyzer with protocol detection, flow reconstruction, and security anomaly detection.
The Prompt:
As a network security engineer, create a Python traffic analyzer using scapy, pyshark, and asyncio with: live packet capture with BPF filtering, protocol identification (HTTP, DNS, TLS, custom), TCP flow reconstruction with reassembly, TLS certificate analysis and JA3 fingerprinting, DDoS detection with entropy analysis, C2 beaconing detection with periodicity analysis, and automatic PCAP generation for suspicious flows. Include visualization with network graphs using pyvis, integration with Suricata for rule matching, and export to Wireshark-compatible formats. Structure as async application with zero-copy packet processing and support for 10Gbps capture rates.


Prompt #20
Title: Document Intelligence Processor
Category: AI Integration
Difficulty: Intermediate
Description: Create an intelligent document processing system that extracts structured data from PDFs, images, and scans.
The Prompt:
Act as an AI document processing specialist. Build a Python system using pytesseract, pdfplumber, and layoutparser with: OCR with deskewing and denoising preprocessing, table extraction with structural preservation, form field detection and value extraction, handwriting recognition with custom model fine-tuning, document classification with layout analysis, named entity recognition for key information, and confidence scoring per extraction. Include support for 10+ languages, redaction capabilities for PII, comparison between document versions, and export to JSON/Excel/Database. Structure as FastAPI service with async processing queue and webhook notifications.


Prompt #21
Title: Python Performance Profiler Suite
Category: Performance Optimization
Difficulty: Advanced
Description: Build a comprehensive profiling toolkit that identifies bottlenecks across CPU, memory, and I/O with actionable recommendations.
The Prompt:
As a Python performance engineer, create a profiling suite using cProfile, memory_profiler, and py-spy with: statistical sampling profiler with minimal overhead, flame graph generation for call stacks, memory allocation tracking with tracemalloc, async/await-aware profiling, database query profiling with SQLAlchemy integration, line-by-line profiling for hot functions, and automatic bottleneck detection with severity scoring. Include comparison mode for A/B performance testing, regression detection in CI/CD, and optimization suggestions with code examples. Structure as VS Code extension, CLI tool, and pytest plugin with comprehensive reporting.


Prompt #22
Title: Smart Home Automation Hub
Category: IoT / Automation
Difficulty: Intermediate
Description: Build a centralized home automation system with device discovery, rule engine, and voice integration.
The Prompt:
Act as an IoT systems developer. Create a Python home automation hub using Home Assistant API, MQTT, and FastAPI with: automatic device discovery via mDNS and SSDP, rule engine with condition-action triggers, energy monitoring with predictive optimization, security system integration with zone-based arming, climate control with ML-based scheduling, voice command processing with Rhasspy integration, and mobile app companion with push notifications. Include 50+ device integrations (lights, sensors, locks, cameras), scene management with gradual transitions, and backup/restore functionality. Structure as modular architecture with plugin system and comprehensive logging.


Prompt #23
Title: Algorithmic Trading Strategy Backtester
Category: Finance Tools
Difficulty: Advanced
Description: Create a vectorized backtesting engine for trading strategies with realistic market simulation.
The Prompt:
As a quantitative developer, build a Python backtesting framework using NumPy, Pandas, and vectorbt with: event-driven and vectorized execution modes, realistic slippage and commission models, position sizing with Kelly criterion, multi-asset portfolio optimization, walk-forward optimization with cross-validation, Monte Carlo simulation for strategy robustness, and regime detection with HMM. Include 20+ built-in technical indicators, custom indicator DSL, and performance metrics (Sortino, Calmar, Omega). Structure as object-oriented framework with comprehensive documentation and example strategies (momentum, mean reversion, pairs trading).


Prompt #24
Title: Malware Analysis Sandbox
Category: Cybersecurity
Difficulty: Expert
Description: Build an automated malware analysis environment with behavioral monitoring and signature generation.
The Prompt:
Assume the role of a malware reverse engineer. Create a Python sandbox using cuckoo, volatility, and yara-python with: Windows API hooking and logging, network traffic capture with IOC extraction, memory dump analysis with process injection detection, string extraction with entropy analysis, YARA rule automatic generation, C2 configuration extraction for common families, and MITRE ATT&CK mapping. Include PDF and Office document analysis with embedded object extraction, Linux ELF analysis support, and integration with VirusTotal and MalwareBazaar. Structure as distributed analysis cluster with REST API and threat intelligence feeds.


Prompt #25
Title: Database Migration Framework
Category: System Design
Difficulty: Intermediate
Description: Create a database schema migration tool with rollback support, data transformation, and multi-environment management.
The Prompt:
Act as a database reliability engineer. Build a Python migration framework using SQLAlchemy, alembic, and jinja2 with: versioned migrations with dependency graph, automatic rollback generation, data transformation scripts with validation hooks, multi-environment configuration (dev/staging/prod), dry-run mode with change preview, online migration support for zero-downtime deployments, and migration performance estimation. Include schema diff visualization, data anonymization for compliance, and integration with CI/CD pipelines. Structure as CLI tool with comprehensive testing utilities and rollback automation.


Prompt #26
Title: Reinforcement Learning Environment Designer
Category: Machine Learning
Difficulty: Expert
Description: Build a framework for creating custom RL environments with curriculum learning and multi-agent support.
The Prompt:
As an RL researcher, create a Python framework using gymnasium, ray, and pytorch with: vectorized environment execution, curriculum learning with automatic difficulty adjustment, multi-agent support with communication protocols, reward shaping debugger with visualization, environment state serialization for reproducibility, and integration with RLlib and Stable-Baselines3. Include 10+ example environments (robotics, trading, games), automatic hyperparameter tuning with Optuna, and comprehensive logging with TensorBoard. Structure as pip-installable package with extensive documentation and benchmark suite.



Prompt #27
Title: API Gateway with Rate Limiting
Category: System Design
Difficulty: Advanced
Description: Build a high-performance API gateway with authentication, rate limiting, and request transformation.
The Prompt:
As a backend architect, create a Python API gateway using aiohttp, redis, and jwt with: JWT validation with JWKS endpoint support, sliding window rate limiting with Redis Lua scripts, request/response transformation with JQ, circuit breaker with half-open state detection, load balancing with health checks, request signing for service mesh, and comprehensive logging with correlation IDs. Include WebSocket proxying, gRPC translation, and OpenAPI aggregation from microservices. Structure as async application with Kubernetes ingress integration and Prometheus metrics.


Prompt #28
Title: Computer Vision Pipeline Builder
Category: Machine Learning
Difficulty: Intermediate
Description: Create a modular computer vision pipeline with preprocessing, augmentation, and model serving.
The Prompt:
Act as a CV engineer. Build a Python pipeline using opencv, albumentations, and triton with: image preprocessing pipeline with auto-optimization, augmentation strategies with bounding box preservation, model versioning with A/B testing, batch inference with dynamic batching, result visualization with Grad-CAM, and performance monitoring with latency percentiles. Include support for classification, detection, segmentation, and OCR tasks, custom operator plugin system, and deployment to edge devices with TensorRT optimization. Structure as configurable YAML-based pipeline with comprehensive debugging tools.


Prompt #29
Title: Blockchain Transaction Monitor
Category: Blockchain
Difficulty: Intermediate
Description: Build a real-time blockchain monitoring system for DeFi protocols with alert generation.
The Prompt:
As a blockchain developer, create a Python monitor using web3.py, eth-brownie, and asyncio with: real-time event log filtering with bloom filters, MEV detection with transaction pattern analysis, whale wallet tracking with clustering, protocol-specific parsing (Uniswap, Aave, Compound), flash loan attack detection heuristics, and Telegram/Discord alert integration. Include portfolio tracking with P&L calculation, impermanent loss monitoring for LPs, and tax reporting export. Structure as async service with PostgreSQL storage and Grafana dashboards.
Prompt #30
Title: Natural Language to Regex Converter
Category: AI Integration
Difficulty: Intermediate
Description: Create a system that converts natural language descriptions into optimized regular expressions.
The Prompt:
Act as a NLP specialist. Build a Python system using transformers, regex, and examples with: few-shot learning from regex pattern library, explanation generation for created patterns, test case generation with positive/negative examples, pattern optimization for performance, and common mistake detection. Include support for Python, JavaScript, and PCRE flavors, visual explanation with regex railroad diagrams, and confidence scoring. Structure as CLI tool and library with comprehensive pattern database and user feedback loop.


Prompt #31
Title: Distributed Lock Manager
Category: System Design
Difficulty: Advanced
Description: Build a distributed locking system with fairness, deadlock detection, and automatic lease renewal.
The Prompt:
As a distributed systems engineer, create a Python lock manager using redis, etcd, or zookeeper with: fair queuing with FIFO ordering, deadlock detection with wait-for graph, automatic lease renewal with jitter, lock downgrade capabilities, and comprehensive metrics. Include fencing token generation for exactly-once processing, hierarchical locking for nested resources, and administrative tools for force-unlock. Structure as async context manager with comprehensive testing including network partition scenarios.


Prompt #32
Title: Time Series Forecasting Platform
Category: Machine Learning
Difficulty: Advanced
Description: Build an end-to-end time series forecasting platform with automatic model selection and ensemble methods.
The Prompt:
Act as a forecasting specialist. Build a Python platform using prophet, sktime, and pytorch-forecasting with: automatic feature engineering (lag, rolling stats, seasonality), model selection with cross-validation, ensemble methods with stacking and blending, uncertainty quantification with conformal prediction, anomaly detection with isolation forest, and automated report generation. Include 15+ models (ARIMA, ETS, N-BEATS, TFT), hyperparameter optimization with Optuna, and deployment to production with MLflow. Structure as configurable pipeline with REST API and batch/real-time modes.


Prompt #33
Title: Code Review Automation Bot
Category: DevOps Scripting
Difficulty: Intermediate
Description: Build a GitHub/GitLab bot that automatically reviews code with intelligent suggestions.
The Prompt:
As a developer experience engineer, create a Python bot using PyGithub, gitlab-python, and openai with: PR summary generation with key changes, style violation detection with ruff/black, security issue flagging with bandit, performance suggestion with complexity analysis, test coverage impact calculation, and constructive comment generation. Include integration with CI pipelines, learning from past reviews, and customizable rule sets per repository. Structure as GitHub App with serverless deployment and comprehensive audit logging.


Prompt #34
Title: ETL Pipeline Orchestrator
Category: Data Engineering
Difficulty: Advanced
Description: Build a modern data pipeline orchestrator with lineage tracking, quality checks, and automatic retries.
The Prompt:
As a data platform engineer, create a Python orchestrator using prefect, pydantic, and great-expectations with: DAG-based pipeline definition with dynamic task generation, data quality validation with automatic quarantine, lineage tracking with OpenLineage, schema evolution handling, incremental processing with watermarking, and backfill automation. Include 30+ connectors (databases, cloud storage, APIs), cost tracking per pipeline, and data freshness SLAs with alerting. Structure as hybrid deployment with local and cloud execution modes.



Prompt #35
Title: Voice Assistant Framework
Category: AI Integration
Difficulty: Expert
Description: Build an offline-capable voice assistant with wake word detection, intent recognition, and skill system.
The Prompt:
As an AI edge computing specialist, create a Python assistant using porcupine, whisper, and rhasspy with: on-device wake word detection, speech-to-text with speaker diarization, intent recognition with slot filling, skill plugin system with manifest validation, text-to-speech with coqui-tts, and conversation state management. Include 20+ built-in skills (weather, timer, smart home), custom skill SDK, and deployment to Raspberry Pi with GPU acceleration. Structure as modular architecture with comprehensive testing and voice profiling.


Prompt #36
Title: Graph Database Query Builder
Category: Data Engineering
Difficulty: Intermediate
Description: Create a type-safe query builder for graph databases with automatic optimization hints.
The Prompt:
Act as a graph database specialist. Build a Python query builder for Neo4j and Amazon Neptune using pydantic with: type-safe node and relationship definitions, query composition with fluent interface, automatic index recommendations, execution plan analysis with hints, result mapping to Pydantic models, and batch operation support. Include Cypher and Gremlin dialects, migration tools for schema changes, and comprehensive query testing. Structure as ORM-like library with async support and connection pooling.


Prompt #37
Title: Load Testing Framework
Category: Performance Optimization
Difficulty: Advanced
Description: Build a distributed load testing framework with realistic user behavior simulation.
The Prompt:
As a performance engineer, create a Python framework using locust, asyncio, and playwright with: scenario-based user behavior definition, think time modeling with distributions, distributed load generation with master-worker, real-time metrics with HDR histograms, automatic bottleneck detection, and chaos engineering integration. Include result comparison across versions, SLA breach alerting, and comprehensive reporting with percentile analysis. Structure as code-based test definition with CLI and CI/CD integration.


Prompt #38
Title: Configuration Management System
Category: DevOps Scripting
Difficulty: Intermediate
Description: Build a hierarchical configuration system with validation, secrets management, and environment overlays.
The Prompt:
Act as a platform engineer. Create a Python configuration system using pydantic, vault, and jinja2 with: YAML/JSON/TOML support with schema validation, environment-specific overlays with inheritance, secret injection with multiple backends, configuration drift detection, and hot-reload capabilities. Include type-safe access with IDE autocomplete, change audit logging, and rollback functionality. Structure as library with CLI for validation and migration tools.



Prompt #39
Title: Automated Penetration Testing Framework
Category: Cybersecurity
Difficulty: Expert
Description: Build an automated pentest framework with reconnaissance, exploitation, and reporting modules.
The Prompt:
As a security automation engineer, create a Python framework using metasploit, nmap, and custom exploits with: subdomain enumeration with permutation, port scanning with service detection, vulnerability scanning with Nuclei integration, automatic exploitation with safety checks, lateral movement simulation, and comprehensive report generation with CVSS scoring. Include network segmentation validation, compliance checking (PCI-DSS, HIPAA), and integration with ticketing systems. Structure as modular framework with strict ethical use guidelines and authorization verification.



Prompt #40
Title: Real-Time Collaboration Server
Category: Networking
Difficulty: Expert
Description: Build a real-time collaborative editing server with operational transformation and presence.
The Prompt:
As a real-time systems engineer, create a Python server using asyncio, websockets, and operational-transform with: OT algorithm for conflict-free editing, cursor presence with awareness, permission system with room management, offline support with sync on reconnect, and scalability with Redis pub/sub. Include rich text and code editing support, version history with branching, and end-to-end encryption option. Structure as async application with comprehensive conflict resolution testing and horizontal scaling support.


Prompt #41
Title: Smart Email Processor
Category: Automation
Difficulty: Intermediate
Description: Build an intelligent email processing system with classification, extraction, and automated responses.
The Prompt:
Act as an automation specialist. Create a Python email processor using imaplib, spacy, and transformers with: automatic classification (urgent, newsletter, support), entity extraction for calendar events, attachment processing with OCR, smart reply generation with tone matching, and automatic forwarding rules. Include spam detection with custom training, thread summarization, and integration with CRM systems. Structure as service with webhook support and comprehensive filtering DSL.


Prompt #42
Title: Quantum Computing Simulator
Category: Scientific Computing
Difficulty: Expert
Description: Build a quantum circuit simulator with noise modeling and algorithm visualization.
The Prompt:
As a quantum computing researcher, create a Python simulator using numpy, numba, and qiskit with: state vector simulation up to 20 qubits, noise channel modeling (depolarizing, amplitude damping), quantum algorithm library (Grover, Shor, VQE), circuit optimization with gate fusion, and interactive Bloch sphere visualization. Include density matrix simulation for mixed states, pulse-level control simulation, and educational tutorials. Structure as Jupyter extension with comprehensive benchmarking against Qiskit Aer.



Prompt #43
Title: Feature Flag Management System
Category: DevOps Scripting
Difficulty: Intermediate
Description: Build a feature flag system with targeting, gradual rollout, and A/B testing integration.
The Prompt:
Act as a release engineer. Create a Python feature flag system using redis, fastapi, and statsd with: user targeting with rule engine, percentage-based rollout with sticky assignment, A/B test integration with automatic significance calculation, kill switches with circuit breaker pattern, and audit logging for all changes. Include SDKs for multiple languages, webhook notifications, and impact analysis dashboards. Structure as SaaS platform with on-premise deployment option.



Prompt #44
Title: 3D Data Visualization Engine
Category: Data Analysis
Difficulty: Advanced
Description: Build an interactive 3D visualization engine for scientific and geospatial data.
The Prompt:
As a scientific visualization expert, create a Python engine using vtk, pyvista, and panel with: volume rendering with transfer functions, isosurface extraction with marching cubes, particle system visualization, geospatial coordinate handling, and time-series animation. Include VR headset support with OpenXR, collaborative annotation, and export to web with WebGL. Structure as library with declarative API and comprehensive performance optimization for large datasets (>10GB).


Prompt #45
Title: Smart Contract Fuzzing Engine
Category: Blockchain
Difficulty: Expert
Description: Build a fuzzing engine for Ethereum smart contracts with coverage-guided mutation.
The Prompt:
As a blockchain security researcher, create a Python fuzzer using eth-brownie, hypothesis, and slither with: coverage-guided mutation with AFL integration, property-based testing with invariant detection, gas usage fuzzing for optimization, reentrancy pattern generation, and automatic exploit minimization. Include EVM state space exploration, symbolic execution hybrid, and integration with CI/CD for regression testing. Structure as comprehensive testing framework with detailed reporting.


Prompt #46
Title: Microservices Testing Harness
Category: Testing & Debugging
Difficulty: Advanced
Description: Build a comprehensive testing framework for microservices with contract testing and chaos engineering.
The Prompt:
As a testing architect, create a Python harness using pytest, pact, and chaosmesh with: consumer-driven contract testing, service virtualization with WireMock, chaos engineering with failure injection, distributed tracing validation, and performance regression detection. Include test data management with synthetic generation, environment parity verification, and comprehensive reporting with risk assessment. Structure as pytest plugin with CI/CD integration and parallel execution support.


Prompt #47
Title: Augmented Reality SDK
Category: Computer Vision
Difficulty: Expert
Description: Build an AR SDK for Python with marker detection, plane estimation, and object occlusion.
The Prompt:
As a computer vision engineer, create a Python AR SDK using opencv, mediapipe, and pyopengl with: marker-based and markerless tracking, plane detection for surface placement, lighting estimation for realistic rendering, object occlusion with depth buffering, and hand gesture recognition for interaction. Include SLAM for world persistence, cloud anchor sharing, and deployment to mobile with Kivy. Structure as library with comprehensive examples and performance optimization for real-time use.


Prompt #48
Title: Biological Sequence Analyzer
Category: Bioinformatics
Difficulty: Advanced
Description: Build a bioinformatics pipeline for DNA/RNA sequence analysis with alignment and variant calling.
The Prompt:
As a bioinformatics specialist, create a Python pipeline using biopython, pysam, and scikit-bio with: sequence alignment with BWA and minimap2, variant calling with freebayes integration, phylogenetic tree construction, motif discovery with MEME, and CRISPR guide RNA design. Include support for FASTA/FASTQ/BAM formats, cloud execution with Cromwell, and visualization with genome browsers. Structure as workflow management system with comprehensive logging and reproducibility guarantees.


Prompt #49
Title: Satellite Image Processor
Category: Geospatial
Difficulty: Advanced
Description: Build a geospatial processing system for satellite imagery with cloud removal and change detection.
The Prompt:
As a remote sensing expert, create a Python system using rasterio, sentinelhub, and pytorch with: atmospheric correction with Sen2Cor, cloud detection and removal with inpainting, change detection with Siamese networks, vegetation index calculation, and object detection for infrastructure. Include time series analysis for deforestation, integration with STAC APIs, and export to GIS formats. Structure as pipeline with cloud-native execution and comprehensive metadata handling.


Prompt #50
Title: Cognitive Architecture Simulator
Category: AI Research
Difficulty: Expert
Description: Build a cognitive architecture implementation with memory systems, reasoning, and learning.
The Prompt:
As an AI researcher, create a Python implementation of a cognitive architecture using soartech, ACT-R principles, and neural networks with: working memory with activation decay, long-term declarative and procedural memory, symbolic reasoning with uncertainty handling, reinforcement learning for skill acquisition, and natural language interface. Include episodic memory with event reconstruction, attention mechanisms for focus, and comprehensive logging for analysis. Structure as research platform with extensible module system and benchmark tasks.



Prompt #51
Title: Smart Farming IoT Platform
Category: IoT / Automation
Difficulty: Intermediate
Description: Build an agricultural monitoring system with sensor integration, irrigation control, and yield prediction.
The Prompt:
Act as an agricultural technology engineer. Create a Python platform using mqtt, influxdb, and scikit-learn with: soil moisture and weather sensor integration, automated irrigation with ET-based scheduling, pest detection with image classification, crop growth modeling with thermal time, and yield prediction with historical data. Include farm management dashboard, equipment tracking with GPS, and integration with drone imagery. Structure as edge-cloud hybrid with offline capability and comprehensive alerting.


Prompt #52
Title: Legal Document Analyzer
Category: NLP / Legal Tech
Difficulty: Advanced
Description: Build an AI system for legal contract analysis with clause extraction, risk assessment, and comparison.
The Prompt:
As a legal technology specialist, create a Python system using spacy, transformers, and legalbert with: contract clause classification (indemnity, termination, governing law), party obligation extraction with deadlines, risk scoring with customizable rules, contract comparison with difference highlighting, and automatic summary generation. Include regulatory compliance checking, precedent case matching, and integration with document management systems. Structure as secure deployment with encryption and audit trails for legal admissibility.



Prompt #53
Title: Network Configuration Generator
Category: Networking
Difficulty: Intermediate
Description: Build a system that generates network device configurations from high-level intent specifications.
The Prompt:
Act as a network automation engineer. Create a Python generator using jinja2, nornir, and yang with: intent-based networking with abstract models, multi-vendor support (Cisco, Juniper, Arista), configuration validation with pyats, drift detection and remediation, and automatic rollback on failure. Include network topology visualization, change impact analysis, and integration with IPAM systems. Structure as CLI and API with comprehensive testing and golden config templates.



Prompt #54
Title: Emotion-Aware Music Generator
Category: Creative AI
Difficulty: Expert
Description: Build a system that generates music based on emotional input with style transfer and real-time adaptation.
The Prompt:
As a creative AI researcher, create a Python system using magenta, pytorch, and librosa with: emotion recognition from text or physiological signals, music generation with transformer models, style transfer between artists, real-time tempo and key adaptation, and MIDI export with virtual instrument rendering. Include interactive composition interface, collaborative filtering for style suggestions, and copyright-clean training data verification. Structure as creative tool with API and standalone application.


Prompt #55
Title: Supply Chain Optimizer
Category: Operations Research
Difficulty: Advanced
Description: Build an optimization system for supply chain with demand forecasting, inventory optimization, and route planning.
The Prompt:
As an operations research specialist, create a Python optimizer using ortools, pulp, and prophet with: demand forecasting with seasonality and promotions, multi-echelon inventory optimization, vehicle routing with time windows, supplier selection with risk scoring, and disruption scenario planning. Include simulation for what-if analysis, carbon footprint optimization, and integration with ERP systems. Structure as optimization service with REST API and comprehensive visualization of solutions.



Prompt #56
Title: Automated UI Testing Framework
Category: Testing & Debugging
Difficulty: Intermediate
Description: Build an intelligent UI testing framework with visual regression, accessibility checks, and self-healing selectors.
The Prompt:
Act as a test automation architect. Create a Python framework using playwright, applitools, and axe-core with: visual regression with AI-powered diffing, accessibility scanning with WCAG compliance, self-healing selectors with ML-based ranking, test generation from user sessions, and parallel execution with sharding. Include flaky test detection with automatic retry analysis, cross-browser matrix testing, and integration with Jira for bug creation. Structure as pytest plugin with comprehensive reporting and CI/CD templates.



Prompt #57
Title: Knowledge Graph Builder
Category: Data Engineering
Difficulty: Advanced
Description: Build an automated knowledge graph construction system from unstructured text with entity linking and relation extraction.
The Prompt:
As a knowledge engineering specialist, create a Python system using spacy, neo4j, and transformers with: named entity recognition with Wikidata linking, relation extraction with distant supervision, coreference resolution for entity consolidation, ontology alignment with schema.org, and graph embedding generation for similarity search. Include incremental updates with change tracking, question answering over the graph, and visualization with D3.js export. Structure as pipeline with configurable components and comprehensive evaluation metrics.



Prompt #58
Title: Robotic Process Automation Engine
Category: Automation
Difficulty: Intermediate
Description: Build an RPA engine with computer vision-based interaction, workflow orchestration, and audit compliance.
The Prompt:
Act as an RPA developer. Create a Python engine using pyautogui, opencv, and airflow with: UI element detection with template matching and OCR, action recording and replay with editing, exception handling with screenshot capture, credential vault integration, and comprehensive audit logging. Include workflow designer with drag-and-drop, bot scheduling with resource management, and ROI analytics dashboard. Structure as enterprise platform with attended and unattended bot support.



Prompt #59
Title: Climate Data Analysis Platform
Category: Scientific Computing
Difficulty: Advanced
Description: Build a platform for climate data analysis with CMIP6 processing, extreme event detection, and projection visualization.
The Prompt:
As a climate scientist, create a Python platform using xarray, dask, and cartopy with: CMIP6 model data processing with bias correction, extreme event detection with peak-over-threshold, climate index calculation (ETCCDI), ensemble statistics with uncertainty quantification, and interactive projection maps. Include downscaling with regional models, impact assessment for sectors, and policy-relevant indicator generation. Structure as Jupyter-based platform with cloud execution and comprehensive data catalog integration.



Prompt #60
Title: Automated Trading Bot Framework
Category: Finance Tools
Difficulty: Expert
Description: Build a comprehensive crypto trading bot framework with strategy backtesting, risk management, and exchange abstraction.
The Prompt:
As a crypto trading systems developer, create a Python framework using ccxt, pandas, and asyncio with: unified exchange API with 100+ exchanges, strategy plugin system with performance metrics, portfolio rebalancing with mean-variance optimization, risk management with VaR and drawdown limits, and social sentiment integration. Include paper trading with slippage simulation, arbitrage detection across exchanges, and tax reporting with FIFO/LIFO. Structure as async framework with comprehensive monitoring and deployment tools.




Prompt #61
Title: Smart City Data Platform
Category: IoT / Smart Cities
Difficulty: Advanced
Description: Build a municipal data platform integrating traffic, energy, waste, and safety systems with predictive analytics.
The Prompt:
Act as a smart city architect. Create a Python platform using kafka, timescaledb, and grafana with: real-time traffic optimization with signal control, energy grid monitoring with demand forecasting, waste collection optimization with route planning, public safety incident prediction, and citizen engagement portal. Include digital twin simulation, open data APIs with privacy protection, and comprehensive sustainability reporting. Structure as microservices with city-specific modules and vendor-agnostic integration.




Prompt #62
Title: Protein Structure Predictor
Category: Bioinformatics
Difficulty: Expert
Description: Build a protein structure prediction pipeline with homology modeling, folding simulation, and quality assessment.
The Prompt:
As a computational biologist, create a Python pipeline using biopython, openmm, and alphafold with: sequence alignment with HHblits, homology modeling with MODELLER integration, ab initio folding with replica exchange, quality assessment with Ramachandran plots, and ligand binding site prediction. Include molecular dynamics refinement, protein-protein docking with HADDOCK, and structure visualization with PyMOL scripting. Structure as workflow system with HPC cluster integration and comprehensive validation metrics.



Prompt #63
Title: Autonomous Drone Controller
Category: Robotics
Difficulty: Expert
Description: Build an autonomous drone control system with SLAM, path planning, and computer vision for navigation.
The Prompt:
As a robotics engineer, create a Python controller using dronekit, ros, and opencv with: GPS-denied navigation with visual SLAM, obstacle avoidance with depth cameras, mission planning with waypoints and geofencing, precision landing with ArUco markers, and swarm coordination with distributed consensus. Include battery-aware mission planning, emergency procedures with RTL, and telemetry logging with blackbox analysis. Structure as ROS package with simulation in Gazebo and hardware-in-the-loop testing.



Prompt #64
Title: Fraud Detection System
Category: Machine Learning
Difficulty: Advanced
Description: Build a real-time fraud detection system with graph analysis, anomaly detection, and explainable AI.
The Prompt:
As a fraud detection specialist, create a Python system using neo4j, sklearn, and shap with: transaction graph analysis with network motifs, anomaly detection with isolation forest and autoencoders, device fingerprinting with entropy analysis, behavioral biometrics with keystroke dynamics, and explainable predictions with LIME. Include real-time scoring with sub-100ms latency, case management workflow, and feedback loop for model improvement. Structure as streaming application with comprehensive monitoring and regulatory compliance.



Prompt #65
Title: Digital Twin Simulator
Category: Industrial IoT
Difficulty: Expert
Description: Build a digital twin framework for industrial equipment with physics-based simulation and predictive maintenance.
The Prompt:
As an industrial digital twin expert, create a Python framework using simpy, fmu, and tensorflow with: physics-based simulation with Modelica FMI integration, sensor data fusion with Kalman filtering, degradation modeling with physics-informed neural networks, predictive maintenance with remaining useful life estimation, and optimization with reinforcement learning. Include 3D visualization with Unity connection, scenario simulation for operator training, and integration with SCADA systems. Structure as modular framework with comprehensive asset modeling tools.



Prompt #66
Title: Personalized Learning Engine
Category: EdTech
Difficulty: Intermediate
Description: Build an adaptive learning system with knowledge tracing, content recommendation, and skill assessment.
The Prompt:
Act as an educational technology engineer. Create a Python engine using django, pytorch, and transformers with: knowledge tracing with deep learning (DKT), content recommendation with collaborative filtering, spaced repetition optimization, automatic question generation with difficulty calibration, and learning path personalization. Include engagement analytics, teacher dashboard with intervention suggestions, and integration with LMS standards (LTI). Structure as web platform with mobile API and comprehensive learning analytics.



Prompt #67
Title: Automated Podcast Producer
Category: Creative AI
Difficulty: Intermediate
Description: Build an automated podcast production system with script generation, voice synthesis, and audio mixing.
The Prompt:
As an audio AI specialist, create a Python production system using gpt-4, elevenlabs, and pydub with: topic research with web scraping, script generation with structure optimization, voice cloning and synthesis with emotion control, audio mixing with music and effects, and automatic publishing to platforms. Include ad insertion with dynamic content, show notes generation with timestamps, and analytics integration. Structure as pipeline with quality control checkpoints and comprehensive audio processing.



Prompt #68
Title: Warehouse Robotics Coordinator
Category: Robotics
Difficulty: Advanced
Description: Build a coordination system for warehouse robots with task allocation, path planning, and collision avoidance.
The Prompt:
As a warehouse automation engineer, create a Python coordinator using ortools, ros, and redis with: multi-agent task allocation with auction algorithms, path planning with A* and dynamic window approach, collision avoidance with velocity obstacles, charging station management, and order batching optimization. Include simulation with realistic physics, integration with WMS systems, and performance analytics with throughput tracking. Structure as distributed system with fault tolerance and comprehensive monitoring.



Prompt #69
Title: Mental Health Monitoring App
Category: Health Tech
Difficulty: Intermediate
Description: Build a mental health monitoring system with mood tracking, crisis detection, and intervention suggestions.
The Prompt:
Act as a digital health engineer. Create a Python backend using fastapi, postgresql, and nlp with: mood tracking with validated scales (PHQ-9, GAD-7), journal analysis with sentiment and topic modeling, crisis detection with risk scoring, intervention suggestions with CBT techniques, and progress visualization with trends. Include privacy-preserving design, clinician dashboard with alerts, and integration with telehealth platforms. Structure as HIPAA-compliant service with comprehensive security and audit trails.



Prompt #70
Title: Automated Code Refactoring Tool
Category: Developer Tools
Difficulty: Advanced
Description: Build an intelligent code refactoring tool with AST manipulation, pattern detection, and safety verification.
The Prompt:
As a developer tools engineer, create a Python refactoring tool using libcst, mypy, and openai with: AST-based transformation with preservation of comments, code smell detection with customizable rules, automated refactoring suggestions with preview, type annotation inference with mypy integration, and safety verification with test impact analysis. Include integration with IDEs (VS Code, PyCharm), batch refactoring with progress tracking, and comprehensive undo support. Structure as CLI and library with extensive testing on popular repositories.



Prompt #71
Title: Renewable Energy Optimizer
Category: Energy Systems
Difficulty: Advanced
Description: Build an optimization system for renewable energy with forecasting, storage management, and grid integration.
The Prompt:
As an energy systems engineer, create a Python optimizer using pyomo, pvlib, and pytorch with: solar/wind forecasting with weather models, battery storage optimization with degradation modeling, demand response with price signals, microgrid control with droop characteristics, and carbon footprint minimization. Include grid code compliance checking, investment planning with NPV analysis, and real-time SCADA integration. Structure as optimization service with comprehensive simulation and deployment tools.



Prompt #72
Title: Augmentative Communication Device
Category: Assistive Tech
Difficulty: Intermediate
Description: Build an AAC system with predictive text, symbol communication, and eye-tracking integration.
The Prompt:
Act as an assistive technology specialist. Create a Python AAC system using tkinter, transformers, and opencv with: word prediction with personal language models, symbol-based communication with customizable boards, text-to-speech with emotion control, eye-tracking cursor control with dwell selection, and switch access for motor impairments. Include vocabulary organization with motor planning, data collection for clinical insights, and cloud backup with privacy protection. Structure as accessible application with comprehensive customization and low-cost hardware support.



Prompt #73
Title: Automated Fact-Checking System
Category: NLP / Misinformation
Difficulty: Expert
Description: Build a fact-checking system with claim detection, evidence retrieval, and veracity prediction.
The Prompt:
As a misinformation researcher, create a Python system using transformers, elasticsearch, and neo4j with: claim detection from text and speech, evidence retrieval from knowledge bases and web, stance detection with neural models, veracity prediction with explanation generation, and source credibility scoring. Include real-time monitoring for trending misinformation, fact-check database management, and integration with social media platforms. Structure as API service with comprehensive evaluation and bias mitigation.



Prompt #74
Title: Smart Grid Cybersecurity Monitor
Category: Cybersecurity / Energy
Difficulty: Expert
Description: Build a cybersecurity monitoring system for power grids with anomaly detection and attack response.
The Prompt:
As a critical infrastructure security engineer, create a Python monitor using scapy, snort, and machine learning with: ICS protocol analysis (DNP3, Modbus, IEC 61850), anomaly detection with behavioral baselines, attack signature matching with MITRE ATT&CK for ICS, automated response with relay control, and forensic data collection. Include compliance with NERC CIP, red team exercise support, and integration with utility control centers. Structure as high-availability system with air-gapped deployment options.



Prompt #75
Title: Personalized Nutrition Advisor
Category: Health Tech
Difficulty: Intermediate
Description: Build a nutrition recommendation system with meal planning, macro tracking, and dietary restriction handling.
The Prompt:
Act as a nutrition technology engineer. Create a Python advisor using fastapi, postgresql, and ortools with: personalized recommendations based on goals and preferences, meal planning with constraint satisfaction, recipe database with nutritional analysis, barcode scanning for logging, and progress tracking with visualization. Include integration with wearable devices, allergen alerting, and grocery list generation with cost optimization. Structure as mobile-backend service with comprehensive food database and scientific backing.



Prompt #76
Title: Automated Video Editor
Category: Creative AI
Difficulty: Advanced
Description: Build an AI-powered video editing system with scene detection, highlight generation, and automatic cutting.
The Prompt:
As a video AI engineer, create a Python editor using moviepy, opencv, and transformers with: scene detection with visual and audio cues, highlight extraction with excitement scoring, automatic cutting with rhythm matching, subtitle generation with translation, and color grading with style transfer. Include template-based editing for common formats, collaborative review with comments, and export optimization for platforms. Structure as pipeline with preview generation and comprehensive format support.



Prompt #77
Title: Distributed Training Orchestrator
Category: Machine Learning Infrastructure
Difficulty: Expert
Description: Build a system for orchestrating distributed deep learning training across heterogeneous hardware.
The Prompt:
As an ML infrastructure engineer, create a Python orchestrator using ray, horovod, and kubernetes with: automatic parallelism strategy selection, fault tolerance with checkpoint resumption, mixed precision training with automatic scaling, hyperparameter search with population-based training, and resource scheduling with gang scheduling. Include support for TPUs, GPUs, and custom accelerators, experiment tracking with weights & biases, and model parallelism with pipeline stages. Structure as kubernetes-native system with comprehensive monitoring and cost optimization.



Prompt #78
Title: Smart Building Energy Manager
Category: IoT / Energy
Difficulty: Intermediate
Description: Build an energy management system for buildings with HVAC optimization, occupancy detection, and demand response.
The Prompt:
Act as a building automation engineer. Create a Python manager using bacnet, influxdb, and scikit-learn with: HVAC optimization with MPC control, occupancy detection with CO2 and PIR sensors, lighting control with daylight harvesting, demand response with load shedding, and energy forecasting with weather integration. Include comfort optimization with PMV calculation, fault detection with FDD algorithms, and integration with BMS systems. Structure as edge-cloud system with comprehensive analytics and certification support.



Prompt #79
Title: Genomic Variant Interpreter
Category: Bioinformatics
Difficulty: Expert
Description: Build a clinical variant interpretation system with annotation, pathogenicity prediction, and reporting.
The Prompt:
As a clinical bioinformatician, create a Python interpreter using pysam, vep, and transformers with: variant annotation with population frequencies, pathogenicity prediction with ensemble models, ACMG guideline automated classification, pharmacogenomic implication checking, and clinical report generation. Include integration with EHR systems, variant sharing with Matchmaker Exchange, and comprehensive audit for regulatory compliance. Structure as HIPAA-compliant service with comprehensive knowledge base integration.




Prompt #80
Title: Autonomous Underwater Vehicle Controller
Category: Robotics
Difficulty: Expert
Description: Build a control system for AUVs with acoustic positioning, mission planning, and scientific payload integration.
The Prompt:
As a marine robotics engineer, create a Python controller using ROS, acoustic positioning libraries, and sensor drivers with: DVL and USBL navigation with Kalman filtering, mission planning with waypoint and lawnmower patterns, adaptive sampling with trigger conditions, scientific payload control (CTD, cameras, samplers), and emergency surfacing with Iridium. Include energy management for long-duration missions, post-mission data processing, and simulation with hydrodynamic models. Structure as ROS-based system with comprehensive testing in water tank and field deployments.




Prompt #81
Title: Personalized Fashion Recommender
Category: E-commerce / ML
Difficulty: Intermediate
Description: Build a fashion recommendation system with style analysis, virtual try-on, and trend forecasting.
The Prompt:
Act as a fashion technology engineer. Create a Python recommender using fastapi, pytorch, and opencv with: style analysis with attribute extraction, outfit compatibility scoring, virtual try-on with GAN-based fitting, trend forecasting with social media analysis, and size recommendation with fit prediction. Include visual search with similarity matching, sustainable fashion scoring, and integration with e-commerce platforms. Structure as API service with mobile SDK and comprehensive analytics.



Prompt #82
Title: Automated Regulatory Compliance Checker
Category: Legal Tech
Difficulty: Advanced
Description: Build a system that checks software projects against regulatory requirements with gap analysis and remediation.
The Prompt:
As a compliance automation engineer, create a Python checker using static analysis, nlp, and knowledge graphs with: requirement extraction from regulatory texts (GDPR, HIPAA, SOX), code pattern matching for compliance controls, documentation completeness checking, audit trail verification, and gap analysis with remediation suggestions. Include continuous monitoring with CI/CD integration, evidence collection for auditors, and cross-regulatory mapping. Structure as enterprise tool with comprehensive reporting and customization.
Prompt #83
Title: Cognitive Behavioral Therapy Chatbot
Category: Health Tech / NLP
Difficulty: Advanced
Description: Build a CBT-based therapeutic chatbot with session management, progress tracking, and crisis escalation.
The Prompt:
As a digital therapeutics engineer, create a Python chatbot using rasa, transformers, and fastapi with: CBT technique implementation (cognitive restructuring, behavioral activation), mood tracking with validated instruments, session structure with homework assignments, progress monitoring with symptom reduction metrics, and crisis detection with human handoff. Include therapist dashboard with supervision tools, personalization with treatment history, and clinical trial integration. Structure as regulated medical device with FDA/IEC 62304 compliance.



Prompt #84
Title: Smart Manufacturing Quality Control
Category: Industrial IoT
Difficulty: Intermediate
Description: Build a computer vision system for manufacturing quality inspection with defect detection and process optimization.
The Prompt:
Act as a manufacturing AI engineer. Create a Python system using opencv, pytorch, and gstreamer with: real-time defect detection with instance segmentation, measurement verification with computer vision, process parameter correlation with quality, predictive maintenance for inspection equipment, and automatic rejection system integration. Include traceability with barcode/QR scanning, SPC chart generation, and integration with MES systems. Structure as edge deployment with cloud analytics and comprehensive validation protocols.



Prompt #85
Title: Astronomical Data Pipeline
Category: Scientific Computing
Difficulty: Expert
Description: Build a data reduction pipeline for astronomical observations with calibration, stacking, and source extraction.
The Prompt:
As an astronomical software engineer, create a Python pipeline using astropy, photutils, and ccdproc with: image calibration with bias/dark/flat, astrometric calibration with Gaia DR3, image stacking with drizzle, source extraction with PSF fitting, and photometry with aperture and PSF. Include spectroscopic reduction with wavelength calibration, time series analysis for exoplanets, and integration with archive systems. Structure as configurable pipeline with HPC execution and comprehensive provenance tracking.



Prompt #86
Title: Autonomous Trading Market Maker
Category: Finance / DeFi
Difficulty: Expert
Description: Build an automated market maker for decentralized exchanges with inventory management and risk controls.
The Prompt:
As a DeFi protocol engineer, create a Python market maker using web3.py, uniswap-sdk, and pandas with: inventory skew management with target ratios, spread optimization with volatility forecasting, toxic flow detection with order pattern analysis, gas optimization with EIP-1559, and hedging with perpetual futures. Include multi-DEX aggregation, MEV protection with private mempool, and comprehensive P&L attribution. Structure as high-frequency trading system with robust error handling and emergency shutdown.



Prompt #87
Title: Intelligent Tutoring System
Category: EdTech / AI
Difficulty: Advanced
Description: Build an ITS with student modeling, hint generation, and misconception remediation.
The Prompt:
As an educational AI researcher, create a Python ITS using django, pytorch, and knowledge tracing with: student skill modeling with Bayesian networks, hint generation with next-step prediction, misconception diagnosis with constraint-based modeling, dialogue management with Socratic questioning, and problem generation with difficulty calibration. Include teacher authoring tools, learning analytics dashboard, and efficacy evaluation with randomized trials. Structure as research platform with extensible domain model and comprehensive logging.




Prompt #88
Title: Smart Waste Management System
Category: Smart Cities / IoT
Difficulty: Intermediate
Description: Build a waste collection optimization system with fill-level monitoring, route planning, and predictive analytics.
The Prompt:
Act as a smart city engineer. Create a Python system using mqtt, ortools, and prophet with: fill-level monitoring with ultrasonic sensors, optimal collection routing with time windows, predictive analytics for container placement, illegal dumping detection with cameras, and recycling contamination analysis. Include citizen app for reporting, fleet management integration, and carbon impact tracking. Structure as municipal platform with comprehensive analytics and cost optimization.
Prompt #89
Title: Quantum Key Distribution Simulator
Category: Cybersecurity / Quantum
Difficulty: Expert
Description: Build a QKD protocol simulator with BB84 and E91 implementations, eavesdropping detection, and error correction.
The Prompt:
As a quantum cryptography researcher, create a Python simulator using qiskit, numpy, and information theory with: BB84 and E91 protocol implementations, photon number splitting attack simulation, error correction with CASCADE, privacy amplification with Toeplitz matrices, and finite key security analysis. Include realistic channel modeling with loss and noise, hardware integration APIs, and comprehensive security proofs. Structure as research tool with visualization and benchmarking against theoretical limits.




Prompt #90
Title: Personalized Medicine Recommender
Category: Health Tech / Bioinformatics
Difficulty: Expert
Description: Build a pharmacogenomic recommendation system with drug-gene interaction checking and dosing optimization.
The Prompt:
As a pharmacogenomic engineer, create a Python system using biopython, knowledge graphs, and clinical guidelines with: CPIC guideline integration for gene-drug pairs, star allele calling from VCF, drug interaction checking with CYP450 enzymes, dosing recommendations with evidence levels, and adverse event prediction. Include EHR integration with FHIR, clinician decision support interface, and regulatory compliance with CLIA. Structure as clinical decision support system with comprehensive knowledge base and audit trails.




Prompt #91
Title: Automated Sports Analytics Platform
Category: Sports Tech / CV
Difficulty: Advanced
Description: Build a sports analytics platform with player tracking, event detection, and performance metrics.
The Prompt:
As a sports technology engineer, create a Python platform using opencv, detectron2, and streamlit with: multi-object tracking with DeepSORT, event detection with pose estimation, tactical analysis with formation recognition, performance metrics with speed and distance, and video synchronization with scoreboard OCR. Include automated highlight generation, scout report generation, and integration with broadcast systems. Structure as platform with real-time and post-game analysis modes.




Prompt #92
Title: Smart Contract Upgradeability Framework
Category: Blockchain
Difficulty: Advanced
Description: Build a framework for secure smart contract upgrades with proxy patterns, storage layout preservation, and migration testing.
The Prompt:
As a blockchain architect, create a Python framework using web3.py, brownie, and slither with: proxy pattern implementation (transparent, UUPS, beacon), storage layout analysis with collision detection, upgrade simulation with state migration, access control with multi-sig integration, and emergency pause functionality. Include automated testing for upgrade safety, gas optimization comparison, and comprehensive documentation. Structure as development framework with CLI and CI/CD integration.




Prompt #93
Title: Automated Penetration Testing Report Generator
Category: Cybersecurity
Difficulty: Intermediate
Description: Build a tool that converts raw pentest data into executive and technical reports with risk scoring.
The Prompt:
Act as a security reporting specialist. Create a Python generator using jinja2, cvss, and matplotlib with: vulnerability deduplication with intelligent matching, risk scoring with CVSS v3.1 and environmental modifiers, executive summary with business impact, technical details with remediation steps, and trend analysis with historical comparison. Include customizable templates, evidence management with screenshots, and integration with Jira/ServiceNow. Structure as CLI tool with comprehensive output formats (PDF, HTML, DOCX).




Prompt #94
Title: Cognitive Load Measurement System
Category: Neuroscience / HCI
Difficulty: Expert
Description: Build a system that measures cognitive load using physiological signals, task performance, and behavioral metrics.
The Prompt:
As a human factors engineer, create a Python system using neurokit2, opencv, and psychopy with: pupillometry for cognitive effort, heart rate variability for stress, EEG processing with band power analysis, task performance metrics with error rates, and behavioral indicators with keystroke dynamics. Include real-time classification with machine learning, adaptive interface recommendations, and integration with eye trackers. Structure as research platform with comprehensive signal processing and validation studies.




Prompt #95
Title: Smart Agriculture Drone System
Category: Robotics / Agriculture
Difficulty: Advanced
Description: Build an agricultural drone system with crop monitoring, precision spraying, and yield estimation.
The Prompt:
As a precision agriculture engineer, create a Python system using dronekit, opencv, and pix4d with: NDVI calculation for crop health, weed detection with semantic segmentation, precision spraying with PWM control, yield estimation with fruit counting, and 3D field mapping with photogrammetry. Include flight planning with terrain following, battery swap automation, and integration with farm management software. Structure as complete UAS solution with ground control station and regulatory compliance.




Prompt #96
Title: Automated Accessibility Auditor
Category: Web Development / Accessibility
Difficulty: Intermediate
Description: Build a comprehensive web accessibility testing tool with WCAG compliance checking and remediation guidance.
The Prompt:
Act as an accessibility engineer. Create a Python auditor using playwright, axe-core, and beautifulsoup with: automated WCAG 2.1 AA testing, color contrast analysis with simulation, screen reader compatibility checking, keyboard navigation verification, and form accessibility validation. Include remediation suggestions with code examples, priority scoring with user impact, and integration with CI/CD pipelines. Structure as CLI and API with comprehensive reporting and trend tracking.




Prompt #97
Title: Personalized News Aggregator
Category: NLP / Information Retrieval
Difficulty: Intermediate
Description: Build an intelligent news aggregation system with bias detection, factuality scoring, and personalization.
The Prompt:
As a news technology engineer, create a Python aggregator using scrapy, transformers, and elasticsearch with: multi-source scraping with normalization, bias detection with political leaning analysis, factuality scoring with source credibility, topic clustering with dynamic categorization, and personalized ranking with explicit feedback. Include filter bubble awareness, diverse viewpoint promotion, and newsletter generation with summaries. Structure as web platform with API and comprehensive content policies.




Prompt #98
Title: Smart Grid Load Balancer
Category: Energy Systems
Difficulty: Expert
Description: Build a load balancing system for power grids with distributed energy resources, demand response, and stability optimization.
The Prompt:
As a power systems engineer, create a Python balancer using pandapower, pyomo, and real-time data with: optimal power flow with distributed generation, demand response aggregation with virtual power plants, frequency regulation with primary/secondary/tertiary control, voltage optimization with reactive power management, and contingency analysis with N-1 criteria. Include stability assessment with transient simulation, market clearing with locational marginal pricing, and integration with SCADA/EMS. Structure as mission-critical system with comprehensive redundancy and regulatory compliance.



Prompt #99
Title: Automated Vulnerability Research Assistant
Category: Cybersecurity Research
Difficulty: Expert
Description: Build an AI assistant for vulnerability research with fuzzing guidance, exploit development, and patch analysis.
The Prompt:
As a vulnerability research engineer, create a Python assistant using angr, ghidra-bridge, and transformers with: binary analysis with symbolic execution, fuzzing campaign guidance with coverage feedback, exploit primitive identification, patch diff analysis for vulnerability location, and PoC generation assistance. Include knowledge base of exploitation techniques, target-specific recommendations, and responsible disclosure workflow management. Structure as research IDE plugin with comprehensive safety guidelines and legal compliance checks.




Prompt #100
Title: Universal Prompt Engineering Framework
Category: AI / Meta
Difficulty: Expert
Description: Build a meta-system that generates optimized prompts for any AI task with automatic refinement and evaluation.
The Prompt:
As a prompt engineering researcher, create a Python framework using langchain, optuna, and evaluation metrics with: prompt template generation with role assignment, automatic few-shot example selection, chain-of-thought optimization, constraint injection for safety, output format specification, and A/B testing with automatic evaluation. Include prompt versioning with diff tracking, performance regression detection, and multi-model optimization (GPT-4, Claude, Llama). Structure as comprehensive SDK with CLI, API, and integration with popular AI frameworks.



