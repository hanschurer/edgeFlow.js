/**
 * edgeFlow.js - Pipeline Exports
 */

import {
  PipelineConfig,
  PipelineTask,
  RuntimeType,
  QuantizationType,
} from '../core/types.js';

// Base
export {
  BasePipeline,
  registerPipeline,
  getPipelineFactory,
  SENTIMENT_LABELS,
  EMOTION_LABELS,
  IMAGENET_LABELS,
  type PipelineResult,
  type TextClassificationResult,
  type FeatureExtractionResult,
  type ImageClassificationResult,
  type ObjectDetectionResult,
} from './base.js';

// Text Classification
export {
  TextClassificationPipeline,
  SentimentAnalysisPipeline,
  createTextClassificationPipeline,
  createSentimentAnalysisPipeline,
  type TextClassificationOptions,
} from './text-classification.js';

// Feature Extraction
export {
  FeatureExtractionPipeline,
  createFeatureExtractionPipeline,
  type FeatureExtractionOptions,
} from './feature-extraction.js';

// Image Classification
export {
  ImageClassificationPipeline,
  createImageClassificationPipeline,
  type ImageClassificationOptions,
  type ImageInput,
} from './image-classification.js';

// ============================================================================
// High-Level Pipeline Factory
// ============================================================================

/**
 * Pipeline options for the factory function
 */
export interface PipelineFactoryOptions {
  /** Model ID or URL */
  model?: string;
  /** Runtime to use */
  runtime?: RuntimeType;
  /** Enable caching */
  cache?: boolean;
  /** Quantization type */
  quantization?: QuantizationType;
  /** Custom labels for classification */
  labels?: string[];
}

/**
 * Supported pipeline task mapping
 */
type PipelineTaskMap = {
  'text-classification': TextClassificationPipeline;
  'sentiment-analysis': SentimentAnalysisPipeline;
  'feature-extraction': FeatureExtractionPipeline;
  'image-classification': ImageClassificationPipeline;
};

// Import pipeline classes
import { TextClassificationPipeline, SentimentAnalysisPipeline } from './text-classification.js';
import { FeatureExtractionPipeline } from './feature-extraction.js';
import { ImageClassificationPipeline } from './image-classification.js';

/**
 * Create a pipeline for a specific task
 * 
 * @example
 * ```typescript
 * // Create a sentiment analysis pipeline
 * const sentiment = await pipeline('sentiment-analysis');
 * const result = await sentiment.run('I love this product!');
 * 
 * // Create an image classifier with custom model
 * const classifier = await pipeline('image-classification', {
 *   model: 'https://example.com/model.bin',
 * });
 * ```
 */
export async function pipeline<T extends keyof PipelineTaskMap>(
  task: T,
  options?: PipelineFactoryOptions
): Promise<PipelineTaskMap[T]> {
  const config: PipelineConfig = {
    task: task as PipelineTask,
    model: options?.model ?? 'default',
    runtime: options?.runtime,
    cache: options?.cache ?? true,
    quantization: options?.quantization,
  };

  let pipelineInstance: TextClassificationPipeline | SentimentAnalysisPipeline | FeatureExtractionPipeline | ImageClassificationPipeline;

  switch (task) {
    case 'text-classification':
      pipelineInstance = new TextClassificationPipeline(config, options?.labels);
      break;
    case 'sentiment-analysis':
      pipelineInstance = new SentimentAnalysisPipeline(config);
      break;
    case 'feature-extraction':
      pipelineInstance = new FeatureExtractionPipeline(config);
      break;
    case 'image-classification':
      pipelineInstance = new ImageClassificationPipeline(config, options?.labels);
      break;
    default:
      throw new Error(`Unknown pipeline task: ${task}`);
  }

  // Initialize the pipeline
  await pipelineInstance.initialize();

  return pipelineInstance as PipelineTaskMap[T];
}

/**
 * Create multiple pipelines at once
 */
export async function createPipelines<T extends (keyof PipelineTaskMap)[]>(
  tasks: T,
  options?: PipelineFactoryOptions
): Promise<{ [K in T[number]]: PipelineTaskMap[K] }> {
  const pipelines = await Promise.all(
    tasks.map(task => pipeline(task, options))
  );

  const result: Partial<{ [K in T[number]]: PipelineTaskMap[K] }> = {};
  
  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i]!;
    result[task as T[number]] = pipelines[i] as PipelineTaskMap[T[number]];
  }

  return result as { [K in T[number]]: PipelineTaskMap[K] };
}
