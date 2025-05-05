"use client"

import type React from "react"

import { useRef } from "react"
import { ImageIcon } from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"

interface TextEditorProps {
  value: string
  onChange: (value: string) => void
  onAddImage: (imageUrl: string) => void
}

export default function TextEditor({ value, onChange, onAddImage }: TextEditorProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      const url = URL.createObjectURL(file)
      onAddImage(url)

      // Reset the input so the same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <div className="relative">
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          className="bg-white/10 border-white/20 hover:bg-white/20 text-white"
        >
          <ImageIcon className="w-4 h-4 mr-2" />
          Add Image
        </Button>
        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
      </div>

      <div className="p-4 pt-16">
        <Textarea
          placeholder="Start writing your research paper here..."
          className="min-h-[300px] bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 resize-none text-white placeholder:text-gray-500"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  )
}
