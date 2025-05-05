"use client"

import { useState } from "react"
import { Lightbulb, Quote, Send, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import TextEditor from "@/components/text-editor"
import CitationPanel from "@/components/citation-panel"
import ContinuationPanel from "@/components/continuation-panel"
import CaptionPanel from "@/components/caption-panel"

export default function ResearchAssistant() {
  const [text, setText] = useState("")
  const [images, setImages] = useState<Array<{ id: string; url: string; caption: string }>>([])
  const [showCitations, setShowCitations] = useState(false)
  const [showContinuations, setShowContinuations] = useState(false)
  const [showCaptions, setShowCaptions] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  const handleTextChange = (value: string) => {
    setText(value)
  }

  const handleAddImage = (imageUrl: string) => {
    const newImage = {
      id: `img-${Date.now()}`,
      url: imageUrl,
      caption: "",
    }
    setImages([...images, newImage])
    setSelectedImage(newImage.id)
    setShowCaptions(true)
    setShowCitations(false)
    setShowContinuations(false)
  }

  const handleRemoveImage = (imageId: string, e: React.MouseEvent) => {
    e.stopPropagation()

    // Filter out the image to remove
    const updatedImages = images.filter((img) => img.id !== imageId)
    setImages(updatedImages)

    if (selectedImage === imageId) {
      if (updatedImages.length > 0) {
        setSelectedImage(updatedImages[0].id)
      } else {
        setSelectedImage(null)
        setShowCaptions(false)
      }
    }

    // Revoke the object URL to free up memory
    const imageToRemove = images.find((img) => img.id === imageId)
    if (imageToRemove) {
      URL.revokeObjectURL(imageToRemove.url)
    }
  }

  const handleImageCaptionChange = (id: string, caption: string) => {
    setImages(images.map((img) => (img.id === id ? { ...img, caption } : img)))
  }

  const handleRequestCitations = () => {
    setShowCitations(true)
    setShowContinuations(false)
    setShowCaptions(false)
  }

  const handleRequestContinuations = () => {
    setShowCitations(false)
    setShowContinuations(true)
    setShowCaptions(false)
  }

  const handleAddCitation = (citation: string) => {
    setText(text + "\n\n" + citation)
    setShowCitations(false)
  }

  const handleAddContinuation = (continuation: string) => {
    setText(continuation)
    setShowContinuations(false)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">
          Research<span className="text-purple-400">Assistant</span>
        </h1>
        <p className="text-gray-300">Your AI-powered research companion</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl overflow-hidden">
            <TextEditor value={text} onChange={handleTextChange} onAddImage={handleAddImage} />

            {images.length > 0 && (
              <div className="border-t border-white/10 p-4">
                <h3 className="text-white font-medium mb-3">Embedded Images</h3>
                <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 md:grid-cols-3">
                  {images.map((image) => (
                    <div
                      key={image.id}
                      //className={`border rounded-lg overflow-hidden ${selectedImage === image.id ? "border-purple-500" : "border-white/10"}`}
                      className={`border rounded-lg overflow-hidden cursor-pointer relative ${
                        selectedImage === image.id ? "border-purple-500" : "border-white/10"
                      }`}
                      onClick={() => {
                        setSelectedImage(image.id)
                        setShowCaptions(true)
                        setShowCitations(false)
                        setShowContinuations(false)
                      }}
                    >
                      <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white rounded-full p-1 z-10"
                      onClick={(e) => handleRemoveImage(image.id, e)}
                    >
                      <Trash2 className="w-4 h-4" />
                      <span className="sr-only">Remove image</span>
                    </Button>
                      <div className="relative h-32">
                        <img
                          src={image.url || "/placeholder.svg"}
                          alt="Research image"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-2 text-xs text-gray-300 truncate">{image.caption || "No caption yet"}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          
            <div className="flex justify-between items-center p-4 border-t border-white/10">
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRequestCitations}
                  className="bg-white/5 border-white/10 hover:bg-white/10 text-white"
                >
                  <Quote className="w-4 h-4 mr-2" />
                  Find Citations
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRequestContinuations}
                  className="bg-white/5 border-white/10 hover:bg-white/10 text-white"
                >
                  <Lightbulb className="w-4 h-4 mr-2" />
                  Continue Writing
                </Button>
              </div>
              <Button size="sm" className="bg-purple-600 hover:bg-purple-700">
                <Send className="w-4 h-4 mr-2" />
                Process
              </Button>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {showCitations && <CitationPanel text={text} onSelectCitation={handleAddCitation} />}

          {showContinuations && <ContinuationPanel text={text} onSelectContinuation={handleAddContinuation} />}

          {showCaptions && selectedImage && (
            <CaptionPanel
              imageUrl={images.find((img) => img.id === selectedImage)?.url || ""}
              onSelectCaption={(caption) => {
                if (selectedImage) {
                  handleImageCaptionChange(selectedImage, caption)
                }
              }}
            />
          )}

          {!showCitations && !showContinuations && !showCaptions && (
            <div className="backdrop-blur-sm bg-white/5 border border-white/10 rounded-xl shadow-xl p-6 text-center text-gray-400">
              <div className="flex flex-col items-center justify-center h-64">
                <Lightbulb className="w-12 h-12 mb-4 text-purple-400 opacity-50" />
                <h3 className="text-xl font-medium text-white mb-2">AI Recommendations</h3>
                <p>Write some text or add an image to get AI-powered suggestions</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
