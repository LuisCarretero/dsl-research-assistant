"use client"

import { Quote, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CitationPanelProps {
  text: string
}

export default function CitationPanel({ text }: CitationPanelProps) {
  // In a real app, these would be generated based on the text
  const citations = [
    {
      id: 1,
      title: "Machine Learning: A Probabilistic Perspective",
      author: "Kevin P. Murphy",
      year: 2012,
      publisher: "MIT Press",
      relevance: 92,
    },
    {
      id: 2,
      title: "Deep Learning",
      author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
      year: 2016,
      publisher: "MIT Press",
      relevance: 87,
    },
    {
      id: 3,
      title: "Artificial Intelligence: A Modern Approach",
      author: "Stuart Russell, Peter Norvig",
      year: 2020,
      publisher: "Pearson",
      relevance: 78,
    },
  ]

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <Quote className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Recommended Citations</h3>
        </div>
        <span className="text-xs text-gray-400">{citations.length} sources found</span>
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {citations.map((citation) => (
          <div key={citation.id} className="p-4 border-b border-white/10 hover:bg-white/5 transition-colors">
            <div className="flex justify-between items-start mb-2">
              <h4 className="font-medium text-white">{citation.title}</h4>
              <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-300">
                {citation.relevance}% match
              </span>
            </div>
            <p className="text-sm text-gray-300 mb-2">
              {citation.author} ({citation.year}), {citation.publisher}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="text-xs bg-white/5 border-white/10 hover:bg-white/10 text-white"
              >
                Insert Citation
              </Button>
              <Button variant="ghost" size="sm" className="text-xs text-gray-400 hover:text-white">
                <ExternalLink className="w-3 h-3 mr-1" />
                View Source
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
