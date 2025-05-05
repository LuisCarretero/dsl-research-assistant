"use client"

import { useState, useEffect } from "react"
import { Lightbulb, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Continuation {
  id: number
  text: string
  confidence: number
}

interface ContinuationPanelProps {
  text: string
  onSelectContinuation?: (text: string) => void
}

export default function ContinuationPanel({ text, onSelectContinuation }: ContinuationPanelProps) {
  const [continuations, setContinuations] = useState<Continuation[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchContinuations = async () => {
      if (!text.trim()) {
        setContinuations([])
        return
      }

      setLoading(true)
      setError(null)

      try {
        // Send to your API
        const response = await fetch("http://localhost:8000/api/generate-continuation/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text }),
        })

        if (!response.ok) {
          throw new Error(`API responded with status: ${response.status}`)
        }

        const data = await response.json()
        setContinuations(data.continuations || [])
      } catch (err) {
        console.error("Error fetching continuations:", err)
        setError("Failed to generate continuations")

        // Fallback to dummy continuations if API fails
        setContinuations([
          {
            id: 1,
            text:
              text +
              " Furthermore, the implementation of neural networks in this context provides a robust framework for analyzing complex patterns within the dataset.",
            confidence: 94,
          },
          {
            id: 2,
            text:
              text +
              " This approach, however, is not without limitations. Several researchers have pointed out potential biases that may emerge from such methodologies.",
            confidence: 87,
          },
          {
            id: 3,
            text:
              text +
              " To address these challenges, we propose a novel algorithm that combines the strengths of both supervised and unsupervised learning techniques.",
            confidence: 82,
          },
        ])
      } finally {
        setLoading(false)
      }
    }

    fetchContinuations()
  }, [text])

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden h-full flex flex-col">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <Lightbulb className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Continue Writing</h3>
        </div>
        <span className="text-xs text-gray-400">
          {loading ? "Generating..." : `${continuations.length} suggestions`}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="p-4 text-center text-gray-400">
            <div className="animate-pulse">Generating continuations...</div>
          </div>
        ) : error ? (
          <div className="p-4 text-center text-red-400">{error}</div>
        ) : continuations.length === 0 ? (
          <div className="p-4 text-center text-gray-400">Start writing to get continuation suggestions</div>
        ) : (
          continuations.map((continuation) => (
            <div key={continuation.id} className="p-4 border-b border-white/10 hover:bg-white/5 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-300">
                  {continuation.confidence}% confidence
                </span>
              </div>
              <p className="text-sm text-white mb-3">
                <span className="text-gray-400">{text.substring(Math.max(0, text.length - 50))}</span>
                <span className="text-purple-300">{continuation.text.substring(text.length)}</span>
              </p>
              <Button
                variant="outline"
                size="sm"
                className="text-xs bg-white/5 border-white/10 hover:bg-white/10 text-white"
                onClick={() => onSelectContinuation && onSelectContinuation(continuation.text)}
              >
                <ArrowRight className="w-3 h-3 mr-1" />
                Use This Continuation
              </Button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
