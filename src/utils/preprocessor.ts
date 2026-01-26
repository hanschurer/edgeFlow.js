/**
 * edgeFlow.js - Preprocessor
 * 
 * Data preprocessing utilities for images, audio, and other data types.
 */

import { EdgeFlowTensor } from '../core/tensor.js';

// ============================================================================
// Image Preprocessing
// ============================================================================

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
 * Default image preprocessing options (ImageNet style)
 */
const DEFAULT_IMAGE_OPTIONS: Required<ImagePreprocessorOptions> = {
  width: 224,
  height: 224,
  resizeMode: 'cover',
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  grayscale: false,
  channelFormat: 'CHW',
  dtype: 'float32',
};

/**
 * ImagePreprocessor - Process images for model input
 */
export class ImagePreprocessor {
  private readonly options: Required<ImagePreprocessorOptions>;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(options: ImagePreprocessorOptions = {}) {
    this.options = { ...DEFAULT_IMAGE_OPTIONS, ...options };
  }

  /**
   * Initialize canvas (lazy)
   */
  private ensureCanvas(): void {
    if (!this.canvas) {
      if (typeof document !== 'undefined') {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
      } else {
        throw new Error('ImagePreprocessor requires a browser environment');
      }
    }
  }

  /**
   * Process an image
   */
  async process(
    input: HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string
  ): Promise<EdgeFlowTensor> {
    let imageData: ImageData;

    if (typeof input === 'string') {
      // Load from URL
      imageData = await this.loadFromUrl(input);
    } else if (input instanceof ImageData) {
      imageData = input;
    } else {
      // Convert to ImageData
      imageData = this.toImageData(input);
    }

    // Resize
    const resized = this.resize(imageData);

    // Convert to tensor
    return this.toTensor(resized);
  }

  /**
   * Process multiple images (batch)
   */
  async processBatch(
    inputs: Array<HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string>
  ): Promise<EdgeFlowTensor> {
    const tensors = await Promise.all(inputs.map(input => this.process(input)));
    
    // Stack tensors into batch
    const batchSize = tensors.length;
    const firstTensor = tensors[0];
    if (!firstTensor) {
      return new EdgeFlowTensor(new Float32Array(0), [0], 'float32');
    }
    
    const channels = firstTensor.shape[0] ?? 3;
    const height = firstTensor.shape[1] ?? this.options.height;
    const width = firstTensor.shape[2] ?? this.options.width;
    
    const batchData = new Float32Array(batchSize * channels * height * width);
    
    for (let i = 0; i < tensors.length; i++) {
      const t = tensors[i];
      if (t) {
        batchData.set(t.toFloat32Array(), i * channels * height * width);
      }
    }

    return new EdgeFlowTensor(
      batchData,
      [batchSize, channels, height, width],
      'float32'
    );
  }

  /**
   * Load image from URL
   */
  private async loadFromUrl(url: string): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        resolve(this.toImageData(img));
      };
      
      img.onerror = () => {
        reject(new Error(`Failed to load image from ${url}`));
      };
      
      img.src = url;
    });
  }

  /**
   * Convert image element to ImageData
   */
  private toImageData(
    source: HTMLImageElement | HTMLCanvasElement | ImageBitmap
  ): ImageData {
    this.ensureCanvas();
    
    const { width, height } = source;
    this.canvas!.width = width;
    this.canvas!.height = height;
    
    this.ctx!.drawImage(source, 0, 0);
    return this.ctx!.getImageData(0, 0, width, height);
  }

  /**
   * Resize image data
   */
  private resize(imageData: ImageData): ImageData {
    const { width, height, resizeMode } = this.options;
    
    this.ensureCanvas();
    
    // Calculate resize dimensions
    let srcX = 0, srcY = 0, srcW = imageData.width, srcH = imageData.height;
    let dstX = 0, dstY = 0, dstW = width, dstH = height;

    if (resizeMode === 'contain') {
      const scale = Math.min(width / imageData.width, height / imageData.height);
      dstW = Math.round(imageData.width * scale);
      dstH = Math.round(imageData.height * scale);
      dstX = Math.round((width - dstW) / 2);
      dstY = Math.round((height - dstH) / 2);
    } else if (resizeMode === 'cover') {
      const scale = Math.max(width / imageData.width, height / imageData.height);
      srcW = Math.round(width / scale);
      srcH = Math.round(height / scale);
      srcX = Math.round((imageData.width - srcW) / 2);
      srcY = Math.round((imageData.height - srcH) / 2);
    }

    // Create temp canvas for source
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCtx.putImageData(imageData, 0, 0);

    // Draw to output canvas
    this.canvas!.width = width;
    this.canvas!.height = height;
    
    // Fill with black for padding modes
    if (resizeMode === 'contain' || resizeMode === 'pad') {
      this.ctx!.fillStyle = 'black';
      this.ctx!.fillRect(0, 0, width, height);
    }
    
    this.ctx!.drawImage(srcCanvas, srcX, srcY, srcW, srcH, dstX, dstY, dstW, dstH);
    
    return this.ctx!.getImageData(0, 0, width, height);
  }

  /**
   * Convert ImageData to tensor
   */
  private toTensor(imageData: ImageData): EdgeFlowTensor {
    const { width, height, mean, std, grayscale, channelFormat, dtype } = this.options;
    const channels = grayscale ? 1 : 3;
    
    const data = new Float32Array(channels * height * width);
    const pixels = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;
        
        if (grayscale) {
          // Convert to grayscale
          const gray = (
            0.299 * (pixels[pixelIdx] ?? 0) +
            0.587 * (pixels[pixelIdx + 1] ?? 0) +
            0.114 * (pixels[pixelIdx + 2] ?? 0)
          ) / 255;
          
          const idx = y * width + x;
          data[idx] = (gray - (mean[0] ?? 0)) / (std[0] ?? 1);
        } else if (channelFormat === 'CHW') {
          // Channel-first format
          for (let c = 0; c < 3; c++) {
            const value = (pixels[pixelIdx + c] ?? 0) / 255;
            const normalized = (value - (mean[c] ?? 0)) / (std[c] ?? 1);
            const idx = c * height * width + y * width + x;
            data[idx] = normalized;
          }
        } else {
          // HWC format
          for (let c = 0; c < 3; c++) {
            const value = (pixels[pixelIdx + c] ?? 0) / 255;
            const normalized = (value - (mean[c] ?? 0)) / (std[c] ?? 1);
            const idx = y * width * 3 + x * 3 + c;
            data[idx] = normalized;
          }
        }
      }
    }

    const shape = channelFormat === 'CHW'
      ? [channels, height, width]
      : [height, width, channels];

    return new EdgeFlowTensor(data, shape, dtype);
  }
}

// ============================================================================
// Audio Preprocessing
// ============================================================================

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
 * Default audio options
 */
const DEFAULT_AUDIO_OPTIONS: Required<AudioPreprocessorOptions> = {
  sampleRate: 16000,
  nMels: 80,
  nFft: 400,
  hopLength: 160,
  normalize: true,
  maxDuration: 30,
};

/**
 * AudioPreprocessor - Process audio for model input
 */
export class AudioPreprocessor {
  private readonly options: Required<AudioPreprocessorOptions>;
  private audioContext: AudioContext | null = null;

  constructor(options: AudioPreprocessorOptions = {}) {
    this.options = { ...DEFAULT_AUDIO_OPTIONS, ...options };
  }

  /**
   * Initialize audio context (lazy)
   */
  private ensureAudioContext(): void {
    if (!this.audioContext) {
      if (typeof AudioContext !== 'undefined') {
        this.audioContext = new AudioContext({ sampleRate: this.options.sampleRate });
      } else {
        throw new Error('AudioPreprocessor requires Web Audio API support');
      }
    }
  }

  /**
   * Process audio data
   */
  async process(input: AudioBuffer | Float32Array | ArrayBuffer | string): Promise<EdgeFlowTensor> {
    let audioData: Float32Array;

    if (typeof input === 'string') {
      // Load from URL
      audioData = await this.loadFromUrl(input);
    } else if (input instanceof AudioBuffer) {
      audioData = this.audioBufferToFloat32(input);
    } else if (input instanceof Float32Array) {
      audioData = input;
    } else {
      // ArrayBuffer - decode
      audioData = await this.decodeAudioData(input);
    }

    // Resample if needed
    // For now, assume input is at target sample rate

    // Normalize
    if (this.options.normalize) {
      audioData = this.normalizeAudio(audioData);
    }

    // Truncate if needed
    const maxSamples = this.options.maxDuration * this.options.sampleRate;
    if (audioData.length > maxSamples) {
      audioData = audioData.slice(0, maxSamples);
    }

    // Compute mel spectrogram (simplified)
    const melSpec = this.computeMelSpectrogram(audioData);

    return melSpec;
  }

  /**
   * Load audio from URL
   */
  private async loadFromUrl(url: string): Promise<Float32Array> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load audio from ${url}`);
    }
    
    const arrayBuffer = await response.arrayBuffer();
    return this.decodeAudioData(arrayBuffer);
  }

  /**
   * Decode audio data
   */
  private async decodeAudioData(data: ArrayBuffer): Promise<Float32Array> {
    this.ensureAudioContext();
    const audioBuffer = await this.audioContext!.decodeAudioData(data);
    return this.audioBufferToFloat32(audioBuffer);
  }

  /**
   * Convert AudioBuffer to Float32Array
   */
  private audioBufferToFloat32(buffer: AudioBuffer): Float32Array {
    // Get first channel
    const channelData = buffer.getChannelData(0);
    return new Float32Array(channelData);
  }

  /**
   * Normalize audio
   */
  private normalizeAudio(data: Float32Array): Float32Array {
    let max = 0;
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i] ?? 0);
      if (abs > max) max = abs;
    }

    if (max > 0) {
      const result = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        result[i] = (data[i] ?? 0) / max;
      }
      return result;
    }

    return data;
  }

  /**
   * Compute mel spectrogram (simplified implementation)
   */
  private computeMelSpectrogram(audio: Float32Array): EdgeFlowTensor {
    const { nMels, nFft, hopLength } = this.options;
    
    // Calculate number of frames
    const numFrames = Math.floor((audio.length - nFft) / hopLength) + 1;
    
    if (numFrames <= 0) {
      // Return empty spectrogram for very short audio
      return new EdgeFlowTensor(new Float32Array(nMels), [1, nMels], 'float32');
    }

    const melSpec = new Float32Array(numFrames * nMels);

    // Simplified mel spectrogram computation
    // In production, use proper FFT and mel filterbank
    for (let frame = 0; frame < numFrames; frame++) {
      const start = frame * hopLength;
      
      // Compute frame energy (simplified - not real FFT)
      for (let mel = 0; mel < nMels; mel++) {
        let energy = 0;
        const freqStart = Math.floor((mel / nMels) * (nFft / 2));
        const freqEnd = Math.floor(((mel + 1) / nMels) * (nFft / 2));
        
        for (let i = freqStart; i < Math.min(freqEnd, nFft); i++) {
          const sample = audio[start + i] ?? 0;
          energy += sample * sample;
        }
        
        // Convert to log scale
        melSpec[frame * nMels + mel] = Math.log(energy + 1e-10);
      }
    }

    return new EdgeFlowTensor(melSpec, [numFrames, nMels], 'float32');
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// ============================================================================
// Text Preprocessing
// ============================================================================

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
export function preprocessText(
  text: string,
  options: TextPreprocessorOptions = {}
): string {
  const {
    lowercase = true,
    removePunctuation = false,
    normalizeWhitespace = true,
    maxLength,
  } = options;

  let result = text;

  if (lowercase) {
    result = result.toLowerCase();
  }

  if (removePunctuation) {
    result = result.replace(/[^\w\s]/g, '');
  }

  if (normalizeWhitespace) {
    result = result.replace(/\s+/g, ' ').trim();
  }

  if (maxLength && result.length > maxLength) {
    result = result.slice(0, maxLength);
  }

  return result;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create image preprocessor with common presets
 */
export function createImagePreprocessor(
  preset: 'imagenet' | 'clip' | 'vit' | 'custom' = 'imagenet',
  options: ImagePreprocessorOptions = {}
): ImagePreprocessor {
  const presets: Record<string, ImagePreprocessorOptions> = {
    imagenet: {
      width: 224,
      height: 224,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
    clip: {
      width: 224,
      height: 224,
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
    },
    vit: {
      width: 224,
      height: 224,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    },
    custom: {},
  };

  return new ImagePreprocessor({ ...presets[preset], ...options });
}

/**
 * Create audio preprocessor with common presets
 */
export function createAudioPreprocessor(
  preset: 'whisper' | 'wav2vec' | 'custom' = 'whisper',
  options: AudioPreprocessorOptions = {}
): AudioPreprocessor {
  const presets: Record<string, AudioPreprocessorOptions> = {
    whisper: {
      sampleRate: 16000,
      nMels: 80,
      nFft: 400,
      hopLength: 160,
    },
    wav2vec: {
      sampleRate: 16000,
      normalize: true,
    },
    custom: {},
  };

  return new AudioPreprocessor({ ...presets[preset], ...options });
}
