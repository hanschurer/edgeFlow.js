/**
 * edgeFlow.js - Tokenizer
 * 
 * Lightweight tokenizer implementation for text processing.
 * Supports BPE, WordPiece, and basic tokenization.
 */

import {
  TokenizerConfig,
  TokenizedOutput,
  EdgeFlowError,
  ErrorCodes,
} from '../core/types.js';

// ============================================================================
// Tokenizer Types
// ============================================================================

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

// ============================================================================
// Base Tokenizer
// ============================================================================

/**
 * Tokenizer - Base class for all tokenizers
 */
export class Tokenizer {
  protected vocab: Map<string, number>;
  protected reverseVocab: Map<number, string>;
  protected config: TokenizerConfig;
  protected model: TokenizerModel;
  protected merges: Map<string, string> = new Map();

  constructor(config: Partial<TokenizerConfig>, options: TokenizerOptions = {}) {
    this.config = {
      vocabSize: config.vocabSize ?? 30522,
      maxLength: config.maxLength ?? 512,
      padTokenId: config.padTokenId ?? 0,
      unkTokenId: config.unkTokenId ?? 100,
      bosTokenId: config.bosTokenId,
      eosTokenId: config.eosTokenId,
      sepTokenId: config.sepTokenId ?? 102,
      clsTokenId: config.clsTokenId ?? 101,
      maskTokenId: config.maskTokenId ?? 103,
    };

    this.model = options.model ?? 'basic';
    this.vocab = new Map();
    this.reverseVocab = new Map();

    // Load vocabulary
    if (options.vocab) {
      this.loadVocab(options.vocab);
    }

    // Load merges for BPE
    if (options.merges) {
      this.loadMerges(options.merges);
    }
  }

  /**
   * Load vocabulary
   */
  protected loadVocab(vocab: Map<string, number> | Record<string, number>): void {
    if (vocab instanceof Map) {
      this.vocab = new Map(vocab);
    } else {
      this.vocab = new Map(Object.entries(vocab));
    }

    // Build reverse vocab
    for (const [token, id] of this.vocab) {
      this.reverseVocab.set(id, token);
    }
  }

  /**
   * Load BPE merges
   */
  protected loadMerges(merges: string[]): void {
    for (const merge of merges) {
      const [a, b] = merge.split(' ');
      if (a && b) {
        this.merges.set(`${a} ${b}`, `${a}${b}`);
      }
    }
  }

  /**
   * Tokenize text
   */
  encode(
    text: string,
    options: {
      addSpecialTokens?: boolean;
      maxLength?: number;
      padding?: 'max_length' | 'longest' | 'do_not_pad';
      truncation?: boolean;
      returnAttentionMask?: boolean;
      returnTokenTypeIds?: boolean;
    } = {}
  ): TokenizedOutput {
    const {
      addSpecialTokens = true,
      maxLength = this.config.maxLength,
      padding = 'max_length',
      truncation = true,
      returnAttentionMask = true,
      returnTokenTypeIds = false,
    } = options;

    // Tokenize
    let tokens = this.tokenize(text);

    // Add special tokens
    if (addSpecialTokens) {
      tokens = this.addSpecialTokens(tokens);
    }

    // Convert to IDs
    let inputIds = this.convertTokensToIds(tokens);

    // Truncate if needed
    if (truncation && inputIds.length > maxLength) {
      inputIds = inputIds.slice(0, maxLength);
      // Ensure EOS token if present
      if (addSpecialTokens && this.config.sepTokenId !== undefined) {
        inputIds[inputIds.length - 1] = this.config.sepTokenId;
      }
    }

    // Create attention mask
    const attentionMask: number[] = returnAttentionMask
      ? inputIds.map(() => 1)
      : [];

    // Pad if needed
    if (padding === 'max_length' && inputIds.length < maxLength) {
      const padLength = maxLength - inputIds.length;
      inputIds = [...inputIds, ...new Array(padLength).fill(this.config.padTokenId) as number[]];
      if (returnAttentionMask) {
        attentionMask.push(...(new Array(padLength).fill(0) as number[]));
      }
    }

    const result: TokenizedOutput = {
      inputIds,
      attentionMask,
    };

    // Token type IDs (for segment embeddings)
    if (returnTokenTypeIds) {
      result.tokenTypeIds = inputIds.map(() => 0);
    }

    return result;
  }

  /**
   * Batch encode
   */
  encodeBatch(
    texts: string[],
    options: {
      addSpecialTokens?: boolean;
      maxLength?: number;
      padding?: 'max_length' | 'longest' | 'do_not_pad';
      truncation?: boolean;
      returnAttentionMask?: boolean;
      returnTokenTypeIds?: boolean;
    } = {}
  ): TokenizedOutput[] {
    // Determine max length for 'longest' padding
    let maxLen = options.maxLength ?? this.config.maxLength;
    
    if (options.padding === 'longest') {
      const encodings = texts.map(text => this.encode(text, { ...options, padding: 'do_not_pad' }));
      maxLen = Math.max(...encodings.map(e => e.inputIds.length));
    }

    return texts.map(text => this.encode(text, { ...options, maxLength: maxLen }));
  }

  /**
   * Decode token IDs back to text
   */
  decode(ids: number[], skipSpecialTokens = true): string {
    const tokens = this.convertIdsToTokens(ids);
    
    // Filter special tokens if requested
    const filteredTokens = skipSpecialTokens
      ? tokens.filter(token => !this.isSpecialToken(token))
      : tokens;

    return this.detokenize(filteredTokens);
  }

  /**
   * Basic tokenization (split by whitespace and punctuation)
   */
  protected tokenize(text: string): string[] {
    // Normalize text
    const normalized = this.normalize(text);

    switch (this.model) {
      case 'bpe':
        return this.tokenizeBPE(normalized);
      case 'wordpiece':
        return this.tokenizeWordPiece(normalized);
      default:
        return this.tokenizeBasic(normalized);
    }
  }

  /**
   * Normalize text
   */
  protected normalize(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s'-]/g, ' $& ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Basic tokenization
   */
  protected tokenizeBasic(text: string): string[] {
    return text.split(/\s+/).filter(t => t.length > 0);
  }

  /**
   * WordPiece tokenization
   */
  protected tokenizeWordPiece(text: string): string[] {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const tokens: string[] = [];

    for (const word of words) {
      const wordTokens = this.tokenizeWord(word);
      tokens.push(...wordTokens);
    }

    return tokens;
  }

  /**
   * Tokenize a single word using WordPiece
   */
  protected tokenizeWord(word: string): string[] {
    if (this.vocab.has(word)) {
      return [word];
    }

    const tokens: string[] = [];
    let start = 0;

    while (start < word.length) {
      let end = word.length;
      let found = false;

      while (start < end) {
        const substr = start === 0 ? word.slice(start, end) : `##${word.slice(start, end)}`;
        
        if (this.vocab.has(substr)) {
          tokens.push(substr);
          found = true;
          break;
        }
        end--;
      }

      if (!found) {
        // Unknown character
        tokens.push('[UNK]');
        start++;
      } else {
        start = end;
      }
    }

    return tokens;
  }

  /**
   * BPE tokenization
   */
  protected tokenizeBPE(text: string): string[] {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const tokens: string[] = [];

    for (const word of words) {
      // Split word into characters
      let chars = word.split('').map((c, i) => i === word.length - 1 ? c + '</w>' : c);

      // Apply merges iteratively
      while (chars.length > 1) {
        let minPair: [number, string] | null = null;
        let minScore = Infinity;

        for (let i = 0; i < chars.length - 1; i++) {
          const pair = `${chars[i]} ${chars[i + 1]}`;
          if (this.merges.has(pair)) {
            const score = Array.from(this.merges.keys()).indexOf(pair);
            if (score < minScore) {
              minScore = score;
              minPair = [i, pair];
            }
          }
        }

        if (!minPair) break;

        const [idx, pair] = minPair;
        const merged = this.merges.get(pair)!;
        chars = [
          ...chars.slice(0, idx),
          merged,
          ...chars.slice(idx + 2),
        ];
      }

      tokens.push(...chars);
    }

    return tokens;
  }

  /**
   * Add special tokens
   */
  protected addSpecialTokens(tokens: string[]): string[] {
    const result: string[] = [];

    // Add CLS token
    if (this.config.clsTokenId !== undefined) {
      result.push('[CLS]');
    }

    result.push(...tokens);

    // Add SEP token
    if (this.config.sepTokenId !== undefined) {
      result.push('[SEP]');
    }

    return result;
  }

  /**
   * Convert tokens to IDs
   */
  protected convertTokensToIds(tokens: string[]): number[] {
    return tokens.map(token => {
      const id = this.vocab.get(token);
      if (id !== undefined) return id;

      // Handle special tokens
      if (token === '[CLS]') return this.config.clsTokenId ?? this.config.unkTokenId;
      if (token === '[SEP]') return this.config.sepTokenId ?? this.config.unkTokenId;
      if (token === '[PAD]') return this.config.padTokenId;
      if (token === '[MASK]') return this.config.maskTokenId ?? this.config.unkTokenId;
      if (token === '[UNK]') return this.config.unkTokenId;

      return this.config.unkTokenId;
    });
  }

  /**
   * Convert IDs to tokens
   */
  protected convertIdsToTokens(ids: number[]): string[] {
    return ids.map(id => {
      const token = this.reverseVocab.get(id);
      if (token !== undefined) return token;

      // Handle special token IDs
      if (id === this.config.clsTokenId) return '[CLS]';
      if (id === this.config.sepTokenId) return '[SEP]';
      if (id === this.config.padTokenId) return '[PAD]';
      if (id === this.config.maskTokenId) return '[MASK]';
      if (id === this.config.unkTokenId) return '[UNK]';

      return '[UNK]';
    });
  }

  /**
   * Check if token is a special token
   */
  protected isSpecialToken(token: string): boolean {
    return ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]'].includes(token);
  }

  /**
   * Detokenize (convert tokens back to text)
   */
  protected detokenize(tokens: string[]): string {
    // Handle WordPiece
    const text = tokens
      .join(' ')
      .replace(/ ##/g, '')
      .replace(/<\/w>/g, ' ')
      .trim();

    return text;
  }

  /**
   * Get vocabulary size
   */
  get vocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Get config
   */
  getConfig(): TokenizerConfig {
    return { ...this.config };
  }
}

// ============================================================================
// Pre-trained Tokenizers
// ============================================================================

/**
 * Create a basic English tokenizer
 */
export function createBasicTokenizer(): Tokenizer {
  // Create basic vocabulary
  const vocab: Record<string, number> = {
    '[PAD]': 0,
    '[UNK]': 1,
    '[CLS]': 2,
    '[SEP]': 3,
    '[MASK]': 4,
  };

  // Add common words
  const commonWords = [
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'as', 'if', 'when', 'while',
    'not', 'no', 'yes', 'all', 'any', 'both', 'each', 'every', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too', 'very',
    'good', 'bad', 'great', 'new', 'old', 'high', 'low', 'big', 'small', 'long', 'short',
    'love', 'like', 'hate', 'want', 'need', 'think', 'know', 'feel', 'see', 'hear',
  ];

  let id = 5;
  for (const word of commonWords) {
    vocab[word] = id++;
  }

  return new Tokenizer(
    {
      vocabSize: id,
      maxLength: 128,
      padTokenId: 0,
      unkTokenId: 1,
      clsTokenId: 2,
      sepTokenId: 3,
      maskTokenId: 4,
    },
    { vocab, model: 'basic' }
  );
}

/**
 * Load tokenizer from URL
 */
export async function loadTokenizer(url: string): Promise<Tokenizer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new EdgeFlowError(
      `Failed to load tokenizer from ${url}`,
      ErrorCodes.MODEL_NOT_FOUND
    );
  }

  const data = await response.json() as {
    config?: Partial<TokenizerConfig>;
    vocab?: Record<string, number>;
    merges?: string[];
    model?: TokenizerModel;
  };

  return new Tokenizer(
    data.config ?? {},
    {
      vocab: data.vocab,
      merges: data.merges,
      model: data.model,
    }
  );
}
