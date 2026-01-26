/**
 * edgeFlow.js - ONNX Runtime Backend
 *
 * Uses onnxruntime-web for real ONNX model inference.
 * Automatically loads ONNX Runtime from CDN when needed.
 */
import { EdgeFlowError, ErrorCodes, } from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';
// ONNX Runtime CDN configuration
const ONNX_VERSION = '1.17.0';
const ONNX_CDN_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/`;
const ONNX_SCRIPT_URL = `${ONNX_CDN_BASE}ort.min.js`;
// Global ONNX Runtime reference (loaded dynamically)
let ort = null;
let ortLoadPromise = null;
/**
 * Dynamically load ONNX Runtime from CDN
 */
async function loadONNXRuntime() {
    // Return cached instance
    if (ort)
        return ort;
    // Return existing load promise to avoid duplicate loading
    if (ortLoadPromise)
        return ortLoadPromise;
    ortLoadPromise = new Promise((resolve, reject) => {
        // Check if already loaded globally (e.g., via script tag)
        if (typeof window !== 'undefined' && window.ort) {
            ort = window.ort;
            // Configure WASM paths
            ort.env.wasm.wasmPaths = ONNX_CDN_BASE;
            resolve(ort);
            return;
        }
        // Dynamically load the script
        const script = document.createElement('script');
        script.src = ONNX_SCRIPT_URL;
        script.async = true;
        script.onload = () => {
            if (window.ort) {
                ort = window.ort;
                // Configure WASM paths
                ort.env.wasm.wasmPaths = ONNX_CDN_BASE;
                console.log(`✓ ONNX Runtime v${ONNX_VERSION} loaded from CDN`);
                resolve(ort);
            }
            else {
                reject(new Error('ONNX Runtime loaded but ort global not found'));
            }
        };
        script.onerror = () => {
            reject(new Error(`Failed to load ONNX Runtime from ${ONNX_SCRIPT_URL}`));
        };
        document.head.appendChild(script);
    });
    return ortLoadPromise;
}
/**
 * Get ONNX Runtime instance (loads if needed)
 */
async function getOrt() {
    if (!ort) {
        ort = await loadONNXRuntime();
    }
    return ort;
}
const sessionStore = new Map();
// ============================================================================
// ONNX Runtime Implementation
// ============================================================================
/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 * Automatically loads ONNX Runtime from CDN when first used.
 */
export class ONNXRuntime {
    name = 'wasm'; // Register as wasm since it's the fallback
    initialized = false;
    executionProvider = 'wasm';
    get capabilities() {
        return {
            concurrency: true,
            quantization: true,
            float16: this.executionProvider === 'webgpu',
            dynamicShapes: true,
            maxBatchSize: 32,
            availableMemory: 512 * 1024 * 1024, // 512MB
        };
    }
    /**
     * Check if ONNX Runtime is available (always true - will be loaded from CDN)
     */
    async isAvailable() {
        // Always return true - we'll load ONNX Runtime from CDN when needed
        return true;
    }
    /**
     * Initialize the ONNX runtime (loads from CDN if needed)
     */
    async initialize() {
        if (this.initialized)
            return;
        // Load ONNX Runtime from CDN
        const ortInstance = await getOrt();
        // Configure WASM paths
        ortInstance.env.wasm.wasmPaths = ONNX_CDN_BASE;
        // Use WASM execution provider (most compatible)
        this.executionProvider = 'wasm';
        this.initialized = true;
    }
    /**
     * Load a model from ArrayBuffer
     */
    async loadModel(modelData, options = {}) {
        if (!this.initialized) {
            await this.initialize();
        }
        const ortInstance = await getOrt();
        try {
            // Create session options
            const sessionOptions = {
                executionProviders: [this.executionProvider],
                graphOptimizationLevel: 'all',
            };
            // Create inference session (convert ArrayBuffer to Uint8Array)
            const modelBytes = new Uint8Array(modelData);
            const session = await ortInstance.InferenceSession.create(modelBytes, sessionOptions);
            // Get input/output names
            const inputNames = session.inputNames;
            const outputNames = session.outputNames;
            // Generate model ID
            const modelId = `onnx_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
            // Store session
            sessionStore.set(modelId, {
                session,
                inputNames: [...inputNames],
                outputNames: [...outputNames],
            });
            // Create metadata
            const metadata = {
                name: options.metadata?.name ?? 'onnx-model',
                version: '1.0.0',
                inputs: inputNames.map(name => ({
                    name,
                    dtype: 'float32',
                    shape: [-1], // Dynamic shape
                })),
                outputs: outputNames.map(name => ({
                    name,
                    dtype: 'float32',
                    shape: [-1],
                })),
                sizeBytes: modelData.byteLength,
                quantization: options.quantization ?? 'float32',
                format: 'onnx',
            };
            // Create model instance
            const model = new LoadedModelImpl(metadata, 'wasm', () => this.unloadModel(modelId));
            // Override the ID to match our stored session
            Object.defineProperty(model, 'id', { value: modelId, writable: false });
            // Track in memory manager
            getMemoryManager().trackModel(model, () => model.dispose());
            return model;
        }
        catch (error) {
            throw new EdgeFlowError(`Failed to load ONNX model: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.MODEL_LOAD_FAILED, { error });
        }
    }
    /**
     * Run inference
     */
    async run(model, inputs) {
        const sessionData = sessionStore.get(model.id);
        if (!sessionData) {
            throw new EdgeFlowError(`ONNX session not found for model ${model.id}`, ErrorCodes.MODEL_NOT_LOADED, { modelId: model.id });
        }
        const ortInstance = await getOrt();
        const { session, inputNames, outputNames } = sessionData;
        try {
            // Prepare input feeds
            const feeds = {};
            for (let i = 0; i < Math.min(inputs.length, inputNames.length); i++) {
                const inputName = inputNames[i];
                const inputTensor = inputs[i];
                if (inputName && inputTensor) {
                    // Convert to ONNX tensor with correct dtype
                    const dtype = inputTensor.dtype;
                    let ortTensor;
                    if (dtype === 'int64') {
                        // Get raw BigInt64Array data directly
                        const data = inputTensor.data;
                        ortTensor = new ortInstance.Tensor('int64', data, inputTensor.shape);
                    }
                    else if (dtype === 'int32') {
                        const data = inputTensor.data;
                        ortTensor = new ortInstance.Tensor('int32', data, inputTensor.shape);
                    }
                    else {
                        const data = inputTensor.toFloat32Array();
                        ortTensor = new ortInstance.Tensor('float32', data, inputTensor.shape);
                    }
                    feeds[inputName] = ortTensor;
                }
            }
            // Run inference
            const results = await session.run(feeds);
            // Convert outputs to EdgeFlowTensor
            const outputs = [];
            for (const outputName of outputNames) {
                const ortTensor = results[outputName];
                if (ortTensor) {
                    const data = ortTensor.data;
                    const shape = Array.from(ortTensor.dims).map(d => Number(d));
                    outputs.push(new EdgeFlowTensor(new Float32Array(data), shape, 'float32'));
                }
            }
            return outputs;
        }
        catch (error) {
            throw new EdgeFlowError(`ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.INFERENCE_FAILED, { modelId: model.id, error });
        }
    }
    /**
     * Unload a model
     */
    async unloadModel(modelId) {
        const sessionData = sessionStore.get(modelId);
        if (sessionData) {
            // Release session will be handled by GC
            sessionStore.delete(modelId);
        }
    }
    /**
     * Dispose the runtime
     */
    dispose() {
        // Clear all sessions
        sessionStore.clear();
        this.initialized = false;
    }
}
/**
 * Create ONNX runtime factory
 */
export function createONNXRuntime() {
    return new ONNXRuntime();
}
//# sourceMappingURL=onnx.js.map