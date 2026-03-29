/**
 * Agent Voice Mapping Spec (RE-MAPPED FOR FREE TIER)
 * These IDs have been verified against your active account using list_voices.ts.
 */

export const AGENT_VOICES = {
  AnalyzeAgent: {
    name: "The Analyst (Bella)",
    voiceId: "hpp4J3VqNfWAUOO0d1Us", // Bella - Professional, Bright, Warm
    settings: { stability: 0.5, similarity_boost: 0.75 }
  },
  DecisionAgent: {
    name: "The Commander (Adam)",
    voiceId: "pNInz6obpgDQGcFmaJgB", // Adam - Dominant, Firm
    settings: { stability: 0.45, similarity_boost: 0.8 }
  },
  ActionAgent: {
    name: "The Enforcer (Charlie)",
    voiceId: "IKne3meq5aSn9XLyUdCD", // Charlie - Deep, Energetic
    settings: { stability: 0.8, similarity_boost: 0.6 }
  },
  ImprovementAgent: {
    name: "The Architect (Matilda)",
    voiceId: "XrExE9yKIg1WjnnlVkGX", // Matilda - Knowledgable, Professional
    settings: { stability: 0.6, similarity_boost: 0.85 }
  },
  System: {
    name: "Core OS (Sarah)",
    voiceId: "EXAVITQu4vr4xnSDxMaL", // Sarah - Mature, Confident
    settings: { stability: 0.5, similarity_boost: 0.5 }
  }
} as const;

export type AgentType = keyof typeof AGENT_VOICES;
