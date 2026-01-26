/**
 * edgeFlow.js - Tokenizer
 *
 * Lightweight tokenizer implementation for text processing.
 * Supports BPE, WordPiece, and basic tokenization.
 */
import { TokenizerConfig, TokenizedOutput } from '../core/types.js';
/**
 * Tokenizer model types
 */
export type TokenizerModel = 'bpe' | 'wordpiece' | 'unigram' | 'basic';
/**
 * Tokenizer options
 */
export interface TokenizerOptions {
    /** Tokenizer model type */
    model?: TokenizerModel;
    /** Vocabulary */
    vocab?: Map<string, number> | Record<string, number>;
    /** Merges for BPE */
    merges?: string[];
    /** Add special tokens */
    addSpecialTokens?: boolean;
    /** Maximum length */
    maxLength?: number;
    /** Padding strategy */
    padding?: 'max_length' | 'longest' | 'do_not_pad';
    /** Truncation */
    truncation?: boolean;
    /** Return attention mask */
    returnAttentionMask?: boolean;
    /** Return token type IDs */
    returnTokenTypeIds?: boolean;
}
/**
 * Tokenizer - Base class for all tokenizers
 */
export declare class Tokenizer {
    protected vocab: Map<string, number>;
    protected reverseVocab: Map<number, string>;
    protected config: TokenizerConfig;
    protected model: TokenizerModel;
    protected merges: Map<string, string>;
    constructor(config: Partial<TokenizerConfig>, options?: TokenizerOptions);
    /**
     * Load vocabulary
     */
    protected loadVocab(vocab: Map<string, number> | Record<string, number>): void;
    /**
     * Load BPE merges
     */
    protected loadMerges(merges: string[]): void;
    /**
     * Tokenize text
     */
    encode(text: string, options?: {
        addSpecialTokens?: boolean;
        maxLength?: number;
        padding?: 'max_length' | 'longest' | 'do_not_pad';
        truncation?: boolean;
        returnAttentionMask?: boolean;
        returnTokenTypeIds?: boolean;
    }): TokenizedOutput;
    /**
     * Batch encode
     */
    encodeBatch(texts: string[], options?: {
        addSpecialTokens?: boolean;
        maxLength?: number;
        padding?: 'max_length' | 'longest' | 'do_not_pad';
        truncation?: boolean;
        returnAttentionMask?: boolean;
        returnTokenTypeIds?: boolean;
    }): TokenizedOutput[];
    /**
     * Decode token IDs back to text
     */
    decode(ids: number[], skipSpecialTokens?: boolean): string;
    /**
     * Basic tokenization (split by whitespace and punctuation)
     */
    protected tokenize(text: string): string[];
    /**
     * Normalize text
     */
    protected normalize(text: string): string;
    /**
     * Basic tokenization
     */
    protected tokenizeBasic(text: string): string[];
    /**
     * WordPiece tokenization
     */
    protected tokenizeWordPiece(text: string): string[];
    /**
     * Tokenize a single word using WordPiece
     */
    protected tokenizeWord(word: string): string[];
    /**
     * BPE tokenization
     */
    protected tokenizeBPE(text: string): string[];
    /**
     * Add special tokens
     */
    protected addSpecialTokens(tokens: string[]): string[];
    /**
     * Convert tokens to IDs
     */
    protected convertTokensToIds(tokens: string[]): number[];
    /**
     * Convert IDs to tokens
     */
    protected convertIdsToTokens(ids: number[]): string[];
    /**
     * Check if token is a special token
     */
    protected isSpecialToken(token: string): boolean;
    /**
     * Detokenize (convert tokens back to text)
     */
    protected detokenize(tokens: string[]): string;
    /**
     * Get vocabulary size
     */
    get vocabSize(): number;
    /**
     * Get config
     */
    getConfig(): TokenizerConfig;
}
/**
 * Create a basic English tokenizer
 */
export declare function createBasicTokenizer(): Tokenizer;
/**
 * Load tokenizer from URL
 */
export declare function loadTokenizer(url: string): Promise<Tokenizer>;
//# sourceMappingURL=tokenizer.d.ts.map