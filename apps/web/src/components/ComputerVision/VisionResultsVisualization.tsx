import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { 
  Eye, 
  Type, 
  Users, 
  Box, 
  Shield, 
  Palette, 
  Crop, 
  Globe, 
  MapPin, 
  Award,
  AlertTriangle,
  Download,
  Copy,
  ExternalLink
} from 'lucide-react';

interface VisionResults {
  labels?: VisionLabel[];
  textDetections?: VisionTextDetection[];
  faces?: VisionFace[];
  objects?: VisionObject[];
  safeSearch?: VisionSafeSearch;
  imageProperties?: VisionImageProperties;
  cropHints?: VisionCropHint[];
  webDetection?: VisionWebDetection;
  landmarks?: VisionLandmark[];
  logos?: VisionLogo[];
  celebrities?: VisionCelebrity[];
  moderationLabels?: VisionModerationLabel[];
  rawResponse?: any;
}

interface VisionLabel {
  name: string;
  confidence: number;
  description?: string;
  category?: string;
  boundingBox?: BoundingBox;
}

interface VisionTextDetection {
  text: string;
  confidence: number;
  language?: string;
  boundingBox?: BoundingBox;
  textType?: string;
}

interface VisionFace {
  boundingBox: BoundingBox;
  confidence: number;
  ageRange?: { low: number; high: number };
  gender?: string;
  emotions?: VisionEmotion[];
  landmarks?: VisionFaceLandmark[];
  attributes?: VisionFaceAttributes;
}

interface VisionObject {
  name: string;
  confidence: number;
  boundingBox: BoundingBox;
  category?: string;
}

interface VisionSafeSearch {
  adult: string;
  spoof: string;
  medical: string;
  violence: string;
  racy: string;
}

interface VisionImageProperties {
  dominantColors: VisionColor[];
  accentColor?: string;
  isBlackAndWhite?: boolean;
  brightness?: number;
  contrast?: number;
  sharpness?: number;
}

interface VisionCropHint {
  boundingBox: BoundingBox;
  confidence: number;
  importanceFraction?: number;
}

interface VisionWebDetection {
  webEntities: VisionWebEntity[];
  fullMatchingImages: VisionWebImage[];
  partialMatchingImages: VisionWebImage[];
  pagesWithMatchingImages: VisionWebPage[];
  visuallySimilarImages: VisionWebImage[];
  bestGuessLabels: string[];
}

interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface VisionEmotion {
  emotionType: string;
  confidence: number;
}

interface VisionFaceLandmark {
  landmarkType: string;
  x: number;
  y: number;
}

interface VisionFaceAttributes {
  smile?: boolean;
  eyeglasses?: boolean;
  sunglasses?: boolean;
  beard?: boolean;
  mustache?: boolean;
  eyesOpen?: boolean;
  mouthOpen?: boolean;
}

interface VisionColor {
  red: number;
  green: number;
  blue: number;
  hexCode: string;
  pixelFraction: number;
  score: number;
}

interface VisionWebEntity {
  entityId?: string;
  description: string;
  score: number;
}

interface VisionWebImage {
  url: string;
  score: number;
}

interface VisionWebPage {
  url: string;
  title?: string;
  score: number;
}

interface VisionLandmark {
  name: string;
  confidence: number;
  boundingBox?: BoundingBox;
  location?: { latitude: number; longitude: number };
}

interface VisionLogo {
  name: string;
  confidence: number;
  boundingBox: BoundingBox;
}

interface VisionCelebrity {
  name: string;
  confidence: number;
  boundingBox: BoundingBox;
  urls?: string[];
  knownGender?: string;
}

interface VisionModerationLabel {
  name: string;
  confidence: number;
  parentName?: string;
}

interface VisionResultsVisualizationProps {
  results: VisionResults;
  imageUrl?: string;
  processingTypes: string[];
}

export const VisionResultsVisualization: React.FC<VisionResultsVisualizationProps> = ({
  results,
  imageUrl,
  processingTypes
}) => {
  const [selectedBoundingBoxes, setSelectedBoundingBoxes] = useState<string[]>([]);

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const getLikelihoodColor = (likelihood: string) => {
    switch (likelihood.toLowerCase()) {
      case 'very_likely': return 'bg-red-500';
      case 'likely': return 'bg-orange-500';
      case 'possible': return 'bg-yellow-500';
      case 'unlikely': return 'bg-blue-500';
      case 'very_unlikely': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `vision-results-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="space-y-6">
      {/* Header with actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Analysis Results</h2>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={downloadResults}>
            <Download className="h-4 w-4 mr-2" />
            Download Results
          </Button>
          <Button variant="outline" size="sm" onClick={() => copyToClipboard(JSON.stringify(results, null, 2))}>
            <Copy className="h-4 w-4 mr-2" />
            Copy JSON
          </Button>
        </div>
      </div>

      {/* Image with overlays */}
      {imageUrl && (
        <Card>
          <CardHeader>
            <CardTitle>Analyzed Image</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative inline-block">
              <img
                src={imageUrl}
                alt="Analyzed"
                className="max-w-full h-auto border rounded-lg"
              />
              {/* Bounding box overlays would be rendered here */}
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="labels" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          {processingTypes.includes('LABEL_DETECTION') && (
            <TabsTrigger value="labels">
              <Eye className="h-4 w-4 mr-2" />
              Labels
            </TabsTrigger>
          )}
          {processingTypes.includes('TEXT_DETECTION') && (
            <TabsTrigger value="text">
              <Type className="h-4 w-4 mr-2" />
              Text
            </TabsTrigger>
          )}
          {processingTypes.includes('FACE_DETECTION') && (
            <TabsTrigger value="faces">
              <Users className="h-4 w-4 mr-2" />
              Faces
            </TabsTrigger>
          )}
          {processingTypes.includes('OBJECT_DETECTION') && (
            <TabsTrigger value="objects">
              <Box className="h-4 w-4 mr-2" />
              Objects
            </TabsTrigger>
          )}
          {processingTypes.includes('SAFE_SEARCH_DETECTION') && (
            <TabsTrigger value="safety">
              <Shield className="h-4 w-4 mr-2" />
              Safety
            </TabsTrigger>
          )}
          {processingTypes.includes('IMAGE_PROPERTIES') && (
            <TabsTrigger value="properties">
              <Palette className="h-4 w-4 mr-2" />
              Properties
            </TabsTrigger>
          )}
        </TabsList>

        {/* Labels Tab */}
        {results.labels && (
          <TabsContent value="labels">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Eye className="h-5 w-5" />
                  <span>Detected Labels ({results.labels.length})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.labels.map((label, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{label.name}</h3>
                        <Badge variant="secondary">{formatConfidence(label.confidence)}</Badge>
                      </div>
                      {label.description && (
                        <p className="text-sm text-muted-foreground mb-2">{label.description}</p>
                      )}
                      {label.category && (
                        <Badge variant="outline" className="text-xs">{label.category}</Badge>
                      )}
                      <Progress value={label.confidence * 100} className="mt-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Text Detection Tab */}
        {results.textDetections && (
          <TabsContent value="text">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Type className="h-5 w-5" />
                  <span>Detected Text ({results.textDetections.length})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.textDetections.map((text, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <p className="font-mono text-sm bg-muted p-2 rounded">{text.text}</p>
                        </div>
                        <div className="ml-4 text-right">
                          <Badge variant="secondary">{formatConfidence(text.confidence)}</Badge>
                          {text.language && (
                            <Badge variant="outline" className="ml-2 text-xs">{text.language}</Badge>
                          )}
                        </div>
                      </div>
                      {text.textType && (
                        <Badge variant="outline" className="text-xs">{text.textType}</Badge>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="mt-2"
                        onClick={() => copyToClipboard(text.text)}
                      >
                        <Copy className="h-3 w-3 mr-1" />
                        Copy
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Face Detection Tab */}
        {results.faces && (
          <TabsContent value="faces">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="h-5 w-5" />
                  <span>Detected Faces ({results.faces.length})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.faces.map((face, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-medium">Face {index + 1}</h3>
                        <Badge variant="secondary">{formatConfidence(face.confidence)}</Badge>
                      </div>
                      
                      <div className="space-y-2 text-sm">
                        {face.ageRange && (
                          <div className="flex justify-between">
                            <span>Age Range:</span>
                            <span>{face.ageRange.low} - {face.ageRange.high}</span>
                          </div>
                        )}
                        {face.gender && (
                          <div className="flex justify-between">
                            <span>Gender:</span>
                            <span>{face.gender}</span>
                          </div>
                        )}
                        
                        {face.attributes && (
                          <div className="mt-3">
                            <h4 className="font-medium mb-2">Attributes:</h4>
                            <div className="flex flex-wrap gap-1">
                              {face.attributes.smile && <Badge variant="outline" className="text-xs">Smiling</Badge>}
                              {face.attributes.eyeglasses && <Badge variant="outline" className="text-xs">Eyeglasses</Badge>}
                              {face.attributes.sunglasses && <Badge variant="outline" className="text-xs">Sunglasses</Badge>}
                              {face.attributes.beard && <Badge variant="outline" className="text-xs">Beard</Badge>}
                              {face.attributes.mustache && <Badge variant="outline" className="text-xs">Mustache</Badge>}
                            </div>
                          </div>
                        )}
                        
                        {face.emotions && face.emotions.length > 0 && (
                          <div className="mt-3">
                            <h4 className="font-medium mb-2">Emotions:</h4>
                            <div className="space-y-1">
                              {face.emotions.slice(0, 3).map((emotion, emotionIndex) => (
                                <div key={emotionIndex} className="flex items-center justify-between">
                                  <span className="text-xs">{emotion.emotionType}</span>
                                  <div className="flex items-center space-x-2">
                                    <Progress value={emotion.confidence * 100} className="w-16 h-2" />
                                    <span className="text-xs">{formatConfidence(emotion.confidence)}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Objects Tab */}
        {results.objects && (
          <TabsContent value="objects">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Box className="h-5 w-5" />
                  <span>Detected Objects ({results.objects.length})</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.objects.map((object, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{object.name}</h3>
                        <Badge variant="secondary">{formatConfidence(object.confidence)}</Badge>
                      </div>
                      {object.category && (
                        <Badge variant="outline" className="text-xs mb-2">{object.category}</Badge>
                      )}
                      <Progress value={object.confidence * 100} className="mt-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Safe Search Tab */}
        {results.safeSearch && (
          <TabsContent value="safety">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Shield className="h-5 w-5" />
                  <span>Safe Search Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(results.safeSearch).map(([category, likelihood]) => (
                    <div key={category} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium capitalize">{category}</h3>
                        <div className={`w-3 h-3 rounded-full ${getLikelihoodColor(likelihood)}`} />
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {likelihood.replace('_', ' ')}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {/* Image Properties Tab */}
        {results.imageProperties && (
          <TabsContent value="properties">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Palette className="h-5 w-5" />
                  <span>Image Properties</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Dominant Colors */}
                  {results.imageProperties.dominantColors && (
                    <div>
                      <h3 className="font-medium mb-3">Dominant Colors</h3>
                      <div className="flex flex-wrap gap-2">
                        {results.imageProperties.dominantColors.slice(0, 10).map((color, index) => (
                          <div key={index} className="flex items-center space-x-2 p-2 border rounded">
                            <div
                              className="w-6 h-6 rounded border"
                              style={{ backgroundColor: color.hexCode }}
                            />
                            <div className="text-xs">
                              <div className="font-mono">{color.hexCode}</div>
                              <div className="text-muted-foreground">
                                {(color.pixelFraction * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Image Quality Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {results.imageProperties.brightness !== undefined && (
                      <div className="p-4 border rounded-lg">
                        <h4 className="font-medium mb-2">Brightness</h4>
                        <Progress value={results.imageProperties.brightness * 100} />
                        <span className="text-sm text-muted-foreground">
                          {(results.imageProperties.brightness * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    {results.imageProperties.contrast !== undefined && (
                      <div className="p-4 border rounded-lg">
                        <h4 className="font-medium mb-2">Contrast</h4>
                        <Progress value={results.imageProperties.contrast * 100} />
                        <span className="text-sm text-muted-foreground">
                          {(results.imageProperties.contrast * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    {results.imageProperties.sharpness !== undefined && (
                      <div className="p-4 border rounded-lg">
                        <h4 className="font-medium mb-2">Sharpness</h4>
                        <Progress value={results.imageProperties.sharpness * 100} />
                        <span className="text-sm text-muted-foreground">
                          {(results.imageProperties.sharpness * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Additional Properties */}
                  <div className="flex flex-wrap gap-2">
                    {results.imageProperties.accentColor && (
                      <Badge variant="outline">
                        Accent: {results.imageProperties.accentColor}
                      </Badge>
                    )}
                    {results.imageProperties.isBlackAndWhite && (
                      <Badge variant="outline">Black & White</Badge>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
};
