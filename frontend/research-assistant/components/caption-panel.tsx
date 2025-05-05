"use client"

import { useState, useEffect } from "react"
import { MessageSquare, Check, Plus } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface Caption {
  id: number
  text: string
  confidence: number
}

interface CaptionPanelProps {
  imageUrl: string
  onSelectCaption?: (caption: string) => void
}

export default function CaptionPanel({ imageUrl, onSelectCaption }: CaptionPanelProps) {
  const [captions, setCaptions] = useState<Caption[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [customCaption, setCustomCaption] = useState("")
  const [showCustomInput, setShowCustomInput] = useState(false)

  useEffect(() => {
    const fetchCaptions = async () => {
      if (!imageUrl) return
      
      setLoading(true)
      setError(null)
      
      try {
        // For blob URLs, we need to fetch the image first
        const response = await fetch(imageUrl)
        const blob = await response.blob()
        
        // Create form data
        const formData = new FormData()
        formData.append('file', blob, 'image.jpg')
        formData.append('prompt', '')
        formData.append('context', '')
        
        // Send to your API
        const captionResponse = await fetch('http://localhost:8000/api/generate-caption/', {
          method: 'POST',
          body: formData,
        })
        
        if (!captionResponse.ok) {
          throw new Error(`API responded with status: ${captionResponse.status}`)
        }
        
        const data = await captionResponse.json()
        setCaptions(data.captions || [])
      } catch (err) {
        console.error('Error fetching captions:', err)
        setError('Failed to generate captions')
      } finally {
        setLoading(false)
      }
    }
    
    fetchCaptions()
  }, [imageUrl])

  const handleSubmitCustomCaption = () => {
    if (customCaption.trim()) {
      if (onSelectCaption) {
        onSelectCaption(customCaption.trim());
      }
      setCustomCaption("")
      setShowCustomInput(false)
    }
  }

  return (
    <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden h-full flex flex-col">
      <div className="p-4 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center">
          <MessageSquare className="w-5 h-5 mr-2 text-purple-400" />
          <h3 className="font-medium text-white">Image Captions</h3>
        </div>
        <span className="text-xs text-gray-400">
          {loading ? 'Generating...' : `${captions.length} suggestions`}
        </span>
      </div>
      
      <div className="flex max-h-[400px] overflow-y-auto">
        {loading ? (
          <div className="p-4 text-center text-gray-400">
            Generating captions...
          </div>
        ) : error ? (
          <div className="p-4 text-center text-red-400">
            {error}
          </div>
        ) : captions.length === 0 ? (
          <div className="p-4 text-center text-gray-400">
            No captions generated
          </div>
        ) : (
          captions.map((caption) => (
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
          ))
        )}

        {/* Custom caption input */}
        {showCustomInput ? (
          <div className="p-4 border-b border-white/10">
            <Textarea
              value={customCaption}
              onChange={(e) => setCustomCaption(e.target.value)}
              placeholder="Enter your own caption..."
              className="mb-3 bg-white/5 border-white/10 text-white"
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="text-xs bg-white/5 border-white/10 hover:bg-white/10 text-white"
                onClick={handleSubmitCustomCaption}
              >
                <Check className="w-3 h-3 mr-1" />
                Use Custom Caption
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-gray-400"
                onClick={() => {
                  setShowCustomInput(false)
                  setCustomCaption("")
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <div className="p-4">
            <Button
              variant="ghost"
              size="sm"
              className="w-full text-xs text-purple-400 hover:text-purple-300 border border-dashed border-white/10 py-3"
              onClick={() => setShowCustomInput(true)}
            >
              <Plus className="w-3 h-3 mr-1" />
              Write Your Own Caption
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}