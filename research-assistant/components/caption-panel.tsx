"use client"

import { MessageSquare, Check } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CaptionPanelProps {
  imageUrl: string
  onSelectCaption?: (caption: string) => void
}

export default function CaptionPanel({ imageUrl, onSelectCaption }: CaptionPanelProps) {
  // In a real app, these would be generated based on the image
  const captions = [
    {
      id: 1,
      text: "Figure 1: Visualization of the neural network architecture used in our experiments, showing input, hidden, and output layers with corresponding activation functions.",
      confidence: 95,
    },
    {
      id: 2,
      text: "Figure 1: Comparison of model performance across different datasets, with the proposed method (in blue) consistently outperforming baseline approaches.",
      confidence: 88,
    },
    {
      id: 3,
      text: "Figure 1: Schematic representation of the experimental setup, including data preprocessing steps and evaluation metrics used in this study.",
      confidence: 81,
    },
  ]

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <MessageSquare className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Image Captions</h3>
        </div>
        <span className="text-xs text-gray-400">{captions.length} suggestions</span>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {captions.map((caption) => (
          <div key={caption.id} className="p-4 border-b border-white/10 hover:bg-white/5 transition-colors">
            <div className="flex justify-between items-start mb-2">
              <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-300">
                {caption.confidence}% confidence
              </span>
            </div>
            <p className="text-sm text-white mb-3">{caption.text}</p>
            <Button
              variant="outline"
              size="sm"
              className="text-xs bg-white/5 border-white/10 hover:bg-white/10 text-white"
              onClick={() => onSelectCaption && onSelectCaption(caption.text)}
            >
              <Check className="w-3 h-3 mr-1" />
              Use This Caption
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}
