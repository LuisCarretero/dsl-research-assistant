"use client"

import { useState, useEffect } from "react"
import { Quote, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Citation {
  id: number
  title: string
  author: string
  year: number
  publisher: string
  relevance: number
  citation: string
}

interface CitationPanelProps {
  text: string
  onSelectCitation?: (citation: string) => void
}

export default function CitationPanel({ text, onSelectCitation }: CitationPanelProps) {
  const [citations, setCitations] = useState<Citation[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchCitations = async () => {
      if (!text.trim()) {
        setCitations([])
        return
      }

      setLoading(true)
      setError(null)

      try {
        // Send to your API
        const response = await fetch("http://localhost:8000/api/generate-citations/", {
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
        setCitations(data.citations || [])
      } catch (err) {
        console.error("Error fetching citations:", err)
        setError("Failed to generate citations")

        setCitations([
          {
            id: 1,
            title: "Citation Title Example",
            author: "",
            year: 2023,
            publisher: "",
            relevance: 92,
            citation: "",
          },
        ])
      } finally {
        setLoading(false)
      }
    }

    fetchCitations()
  }, [text])

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden h-full flex flex-col">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <Quote className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Recommended Citations</h3>
        </div>
        <span className="text-xs text-gray-400">{loading ? "Searching..." : `${citations.length} sources found`}</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="p-4 text-center text-gray-400">
            <div className="animate-pulse">Searching for relevant citations...</div>
          </div>
        ) : error ? (
          <div className="p-4 text-center text-red-400">{error}</div>
        ) : citations.length === 0 ? (
          <div className="p-4 text-center text-gray-400">Write more text to get citation recommendations</div>
        ) : (
          citations.map((citation) => (
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
                  onClick={() => onSelectCitation && onSelectCitation(citation.citation)}
                >
                  Insert Citation
                </Button>
                <Button variant="ghost" size="sm" className="text-xs text-gray-400 hover:text-white">
                  <ExternalLink className="w-3 h-3 mr-1" />
                  View Source
                </Button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
