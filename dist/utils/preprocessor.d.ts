/**
 * edgeFlow.js - Preprocessor
 *
 * Data preprocessing utilities for images, audio, and other data types.
 */
import { EdgeFlowTensor } from '../core/tensor.js';
/**
 * Image preprocessing options
 */
export interface ImagePreprocessorOptions {
    /** Target width */
    width?: number;
    /** Target height */
    height?: number;
    /** Resize mode */
    resizeMode?: 'stretch' | 'contain' | 'cover' | 'pad';
    /** Normalization mean */
    mean?: [number, number, number];
    /** Normalization std */
    std?: [number, number, number];
    /** Convert to grayscale */
    grayscale?: boolean;
    /** Channel format */
    channelFormat?: 'CHW' | 'HWC';
    /** Output data type */
    dtype?: 'float32' | 'uint8';
}
/**
 * ImagePreprocessor - Process images for model input
 */
export declare class ImagePreprocessor {
    private readonly options;
    private canvas;
    private ctx;
    constructor(options?: ImagePreprocessorOptions);
    /**
     * Initialize canvas (lazy)
     */
    private ensureCanvas;
    /**
     * Process an image
     */
    process(input: HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string): Promise<EdgeFlowTensor>;
    /**
     * Process multiple images (batch)
     */
    processBatch(inputs: Array<HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string>): Promise<EdgeFlowTensor>;
    /**
     * Load image from URL
     */
    private loadFromUrl;
    /**
     * Convert image element to ImageData
     */
    private toImageData;
    /**
     * Resize image data
     */
    private resize;
    /**
     * Convert ImageData to tensor
     */
    private toTensor;
}
/**
 * Audio preprocessing options
 */
export interface AudioPreprocessorOptions {
    /** Target sample rate */
    sampleRate?: number;
    /** Number of mel bins */
    nMels?: number;
    /** FFT size */
    nFft?: number;
    /** Hop length */
    hopLength?: number;
    /** Whether to normalize */
    normalize?: boolean;
    /** Maximum duration in seconds */
    maxDuration?: number;
}
/**
 * AudioPreprocessor - Process audio for model input
 */
export declare class AudioPreprocessor {
    private readonly options;
    private audioContext;
    constructor(options?: AudioPreprocessorOptions);
    /**
     * Initialize audio context (lazy)
     */
    private ensureAudioContext;
    /**
     * Process audio data
     */
    process(input: AudioBuffer | Float32Array | ArrayBuffer | string): Promise<EdgeFlowTensor>;
    /**
     * Load audio from URL
     */
    private loadFromUrl;
    /**
     * Decode audio data
     */
    private decodeAudioData;
    /**
     * Convert AudioBuffer to Float32Array
     */
    private audioBufferToFloat32;
    /**
     * Normalize audio
     */
    private normalizeAudio;
    /**
     * Compute mel spectrogram (simplified implementation)
     */
    private computeMelSpectrogram;
    /**
     * Dispose resources
     */
    dispose(): void;
}
/**
 * Text preprocessing options
 */
export interface TextPreprocessorOptions {
    /** Convert to lowercase */
    lowercase?: boolean;
    /** Remove punctuation */
    removePunctuation?: boolean;
    /** Remove extra whitespace */
    normalizeWhitespace?: boolean;
    /** Maximum length in characters */
    maxLength?: number;
}
/**
 * Preprocess text
 */
export declare function preprocessText(text: string, options?: TextPreprocessorOptions): string;
/**
 * Create image preprocessor with common presets
 */
export declare function createImagePreprocessor(preset?: 'imagenet' | 'clip' | 'vit' | 'custom', options?: ImagePreprocessorOptions): ImagePreprocessor;
/**
 * Create audio preprocessor with common presets
 */
export declare function createAudioPreprocessor(preset?: 'whisper' | 'wav2vec' | 'custom', options?: AudioPreprocessorOptions): AudioPreprocessor;
//# sourceMappingURL=preprocessor.d.ts.map