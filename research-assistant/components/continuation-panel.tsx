"use client"

import { Lightbulb, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ContinuationPanelProps {
  text: string
}

export default function ContinuationPanel({ text }: ContinuationPanelProps) {
  // In a real app, these would be generated based on the text
  const continuations = [
    {
      id: 1,
      text: "Furthermore, the implementation of neural networks in this context provides a robust framework for analyzing complex patterns within the dataset.",
      confidence: 94,
    },
    {
      id: 2,
      text: "This approach, however, is not without limitations. Several researchers have pointed out potential biases that may emerge from such methodologies.",
      confidence: 87,
    },
    {
      id: 3,
      text: "To address these challenges, we propose a novel algorithm that combines the strengths of both supervised and unsupervised learning techniques.",
      confidence: 82,
    },
  ]

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <Lightbulb className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Continue Writing</h3>
        </div>
        <span className="text-xs text-gray-400">{continuations.length} suggestions</span>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {continuations.map((continuation) => (
          <div key={continuation.id} className="p-4 border-b border-white/10 hover:bg-white/5 transition-colors">
            <div className="flex justify-between items-start mb-2">
              <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-300">
                {continuation.confidence}% confidence
              </span>
            </div>
            <p className="text-sm text-white mb-3">{continuation.text}</p>
            <Button
              variant="outline"
              size="sm"
              className="text-xs bg-white/5 border-white/10 hover:bg-white/10 text-white"
            >
              <ArrowRight className="w-3 h-3 mr-1" />
              Use This Continuation
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
