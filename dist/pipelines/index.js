/**
 * edgeFlow.js - Pipeline Exports
 */
// Base
export { BasePipeline, registerPipeline, getPipelineFactory, SENTIMENT_LABELS, EMOTION_LABELS, IMAGENET_LABELS, } from './base.js';
// Text Classification
export { TextClassificationPipeline, SentimentAnalysisPipeline, createTextClassificationPipeline, createSentimentAnalysisPipeline, } from './text-classification.js';
// Feature Extraction
export { FeatureExtractionPipeline, createFeatureExtractionPipeline, } from './feature-extraction.js';
// Image Classification
export { ImageClassificationPipeline, createImageClassificationPipeline, } from './image-classification.js';
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
export async function pipeline(task, options) {
    const config = {
        task: task,
        model: options?.model ?? 'default',
        runtime: options?.runtime,
        cache: options?.cache ?? true,
        quantization: options?.quantization,
    };
    let pipelineInstance;
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
    return pipelineInstance;
}
/**
 * Create multiple pipelines at once
 */
export async function createPipelines(tasks, options) {
    const pipelines = await Promise.all(tasks.map(task => pipeline(task, options)));
    const result = {};
    for (let i = 0; i < tasks.length; i++) {
        const task = tasks[i];
        result[task] = pipelines[i];
    }
    return result;
}
//# sourceMappingURL=index.js.map