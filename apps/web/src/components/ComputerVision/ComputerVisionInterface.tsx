import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, Eye, Brain, Search, Shield, Palette, Crop, Globe, MapPin, Award, AlertTriangle, DollarSign } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Types for computer vision
interface VisionProvider {
  provider: string;
  displayName: string;
  isAvailable: boolean;
  supportedTypes: string[];
}

interface VisionProcessingRequest {
  imageUrl?: string;
  imageData?: string;
  processingTypes: string[];
  provider?: string;
  options?: {
    maxResults?: number;
    minConfidence?: number;
    language?: string;
    includeGeoResults?: boolean;
  };
}

interface VisionProcessingJob {
  id: string;
  userId: string;
  imageUrl?: string;
  processingType: string;
  provider: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  results?: any;
  errorMessage?: string;
  processingTimeMs?: number;
  costCents?: number;
  createdAt: string;
  completedAt?: string;
}

interface CostStats {
  dailyCostCents: number;
  monthlyCostCents: number;
  dailyLimitCents: number;
  monthlyLimitCents: number;
}

const PROCESSING_TYPES = [
  { value: 'LABEL_DETECTION', label: 'Label Detection', icon: Eye, description: 'Identify objects, animals, and concepts' },
  { value: 'TEXT_DETECTION', label: 'Text Detection', icon: Search, description: 'Extract text from images (OCR)' },
  { value: 'FACE_DETECTION', label: 'Face Detection', icon: Brain, description: 'Detect faces and facial attributes' },
  { value: 'OBJECT_DETECTION', label: 'Object Detection', icon: Eye, description: 'Locate and identify objects' },
  { value: 'SAFE_SEARCH_DETECTION', label: 'Safe Search', icon: Shield, description: 'Detect inappropriate content' },
  { value: 'IMAGE_PROPERTIES', label: 'Image Properties', icon: Palette, description: 'Analyze colors and properties' },
  { value: 'CROP_HINTS', label: 'Crop Hints', icon: Crop, description: 'Suggest optimal crop regions' },
  { value: 'WEB_DETECTION', label: 'Web Detection', icon: Globe, description: 'Find similar images on the web' },
  { value: 'LANDMARK_DETECTION', label: 'Landmark Detection', icon: MapPin, description: 'Identify famous landmarks' },
  { value: 'LOGO_DETECTION', label: 'Logo Detection', icon: Award, description: 'Detect brand logos' },
  { value: 'CELEBRITY_RECOGNITION', label: 'Celebrity Recognition', icon: Award, description: 'Recognize celebrities' },
  { value: 'MODERATION_LABELS', label: 'Content Moderation', icon: AlertTriangle, description: 'Moderate content for safety' },
];

export const ComputerVisionInterface: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [selectedTypes, setSelectedTypes] = useState<string[]>(['LABEL_DETECTION']);
  const [providers, setProviders] = useState<VisionProvider[]>([]);
  const [processing, setProcessing] = useState(false);
  const [currentJob, setCurrentJob] = useState<VisionProcessingJob | null>(null);
  const [jobHistory, setJobHistory] = useState<VisionProcessingJob[]>([]);
  const [costStats, setCostStats] = useState<CostStats | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [maxResults, setMaxResults] = useState<number>(10);
  const [minConfidence, setMinConfidence] = useState<number>(50);
  const { toast } = useToast();

  // Load available providers on component mount
  useEffect(() => {
    loadAvailableProviders();
    loadCostStats();
    loadJobHistory();
  }, []);

  const loadAvailableProviders = async () => {
    try {
      // This would call the Tauri command
      // const providers = await invoke('get_available_vision_providers');
      // For now, mock data
      const mockProviders: VisionProvider[] = [
        {
          provider: 'GOOGLE_VISION',
          displayName: 'Google Vision API',
          isAvailable: true,
          supportedTypes: ['LABEL_DETECTION', 'TEXT_DETECTION', 'FACE_DETECTION', 'OBJECT_DETECTION', 'SAFE_SEARCH_DETECTION', 'IMAGE_PROPERTIES', 'CROP_HINTS', 'WEB_DETECTION', 'LANDMARK_DETECTION', 'LOGO_DETECTION']
        },
        {
          provider: 'AWS_REKOGNITION',
          displayName: 'AWS Rekognition',
          isAvailable: true,
          supportedTypes: ['LABEL_DETECTION', 'TEXT_DETECTION', 'FACE_DETECTION', 'CELEBRITY_RECOGNITION', 'MODERATION_LABELS', 'OBJECT_DETECTION']
        },
        {
          provider: 'AZURE_COMPUTER_VISION',
          displayName: 'Azure Computer Vision',
          isAvailable: true,
          supportedTypes: ['LABEL_DETECTION', 'TEXT_DETECTION', 'FACE_DETECTION', 'OBJECT_DETECTION', 'IMAGE_ANALYSIS', 'IMAGE_PROPERTIES']
        }
      ];
      setProviders(mockProviders);
      if (mockProviders.length > 0) {
        setSelectedProvider(mockProviders[0].provider);
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load available providers',
        variant: 'destructive',
      });
    }
  };

  const loadCostStats = async () => {
    try {
      // This would call the Tauri command
      // const stats = await invoke('get_vision_cost_stats');
      // For now, mock data
      const mockStats: CostStats = {
        dailyCostCents: 150,
        monthlyCostCents: 2500,
        dailyLimitCents: 1000,
        monthlyLimitCents: 10000,
      };
      setCostStats(mockStats);
    } catch (error) {
      console.error('Failed to load cost stats:', error);
    }
  };

  const loadJobHistory = async () => {
    try {
      // This would call the Tauri command
      // const history = await invoke('get_user_vision_history', { userId: 'current-user', limit: 10 });
      // For now, mock data
      const mockHistory: VisionProcessingJob[] = [];
      setJobHistory(mockHistory);
    } catch (error) {
      console.error('Failed to load job history:', error);
    }
  };

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: 'File too large',
          description: 'Please select an image smaller than 10MB',
          variant: 'destructive',
        });
        return;
      }

      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
      if (!allowedTypes.includes(file.type)) {
        toast({
          title: 'Invalid file type',
          description: 'Please select a valid image file (JPEG, PNG, GIF, BMP, WebP)',
          variant: 'destructive',
        });
        return;
      }

      setSelectedFile(file);
      setImageUrl(''); // Clear URL if file is selected
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  }, [toast]);

  const handleUrlChange = useCallback((url: string) => {
    setImageUrl(url);
    setSelectedFile(null); // Clear file if URL is provided
    setPreviewUrl(url);
  }, []);

  const handleProcessingTypeToggle = useCallback((type: string, checked: boolean) => {
    if (checked) {
      setSelectedTypes(prev => [...prev, type]);
    } else {
      setSelectedTypes(prev => prev.filter(t => t !== type));
    }
  }, []);

  const getAvailableTypesForProvider = useCallback((providerName: string) => {
    const provider = providers.find(p => p.provider === providerName);
    return provider?.supportedTypes || [];
  }, [providers]);

  const processImage = async () => {
    if (!selectedFile && !imageUrl) {
      toast({
        title: 'No image selected',
        description: 'Please select an image file or provide an image URL',
        variant: 'destructive',
      });
      return;
    }

    if (selectedTypes.length === 0) {
      toast({
        title: 'No processing types selected',
        description: 'Please select at least one processing type',
        variant: 'destructive',
      });
      return;
    }

    setProcessing(true);
    
    try {
      let imageData: string | undefined;
      
      if (selectedFile) {
        // Convert file to base64
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            const result = reader.result as string;
            // Remove data URL prefix
            const base64Data = result.split(',')[1];
            resolve(base64Data);
          };
          reader.onerror = reject;
          reader.readAsDataURL(selectedFile);
        });
        imageData = base64;
      }

      const request: VisionProcessingRequest = {
        imageUrl: imageUrl || undefined,
        imageData,
        processingTypes: selectedTypes,
        provider: selectedProvider || undefined,
        options: {
          maxResults,
          minConfidence: minConfidence / 100, // Convert percentage to decimal
          language: 'en',
          includeGeoResults: false,
        },
      };

      // This would call the Tauri command
      // const response = await invoke('process_image', { 
      //   userId: 'current-user', 
      //   request 
      // });
      
      // Mock response for development
      const mockResponse = {
        jobId: `job-${Date.now()}`,
        status: 'PROCESSING',
        providerUsed: selectedProvider,
      };

      setCurrentJob({
        id: mockResponse.jobId,
        userId: 'current-user',
        imageUrl: imageUrl || undefined,
        processingType: selectedTypes[0],
        provider: selectedProvider,
        status: 'PROCESSING',
        createdAt: new Date().toISOString(),
      });

      toast({
        title: 'Processing started',
        description: `Image processing job ${mockResponse.jobId} has been started`,
      });

      // Poll for job status (in real implementation)
      // pollJobStatus(mockResponse.jobId);
      
    } catch (error) {
      toast({
        title: 'Processing failed',
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: 'destructive',
      });
    } finally {
      setProcessing(false);
    }
  };

  const formatCost = (cents: number) => {
    return `$${(cents / 100).toFixed(2)}`;
  };

  const getProcessingTypeIcon = (type: string) => {
    const typeInfo = PROCESSING_TYPES.find(t => t.value === type);
    return typeInfo?.icon || Eye;
  };

  const selectedProviderInfo = providers.find(p => p.provider === selectedProvider);
  const availableTypes = selectedProviderInfo?.supportedTypes || [];

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Computer Vision</h1>
          <p className="text-muted-foreground">
            Analyze images using advanced AI vision models
          </p>
        </div>
        {costStats && (
          <Card className="w-64">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4" />
                <div className="text-sm">
                  <div>Daily: {formatCost(costStats.dailyCostCents)} / {formatCost(costStats.dailyLimitCents)}</div>
                  <div>Monthly: {formatCost(costStats.monthlyCostCents)} / {formatCost(costStats.monthlyLimitCents)}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <Tabs defaultValue="process" className="space-y-6">
        <TabsList>
          <TabsTrigger value="process">Process Image</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="process" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Image Input */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Upload className="h-5 w-5" />
                  <span>Image Input</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="image-file">Upload Image File</Label>
                  <Input
                    id="image-file"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="mt-1"
                  />
                </div>
                
                <div className="text-center text-muted-foreground">or</div>
                
                <div>
                  <Label htmlFor="image-url">Image URL</Label>
                  <Input
                    id="image-url"
                    type="url"
                    placeholder="https://example.com/image.jpg"
                    value={imageUrl}
                    onChange={(e) => handleUrlChange(e.target.value)}
                    className="mt-1"
                  />
                </div>

                {previewUrl && (
                  <div className="mt-4">
                    <Label>Preview</Label>
                    <div className="mt-2 border rounded-lg overflow-hidden">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="w-full h-48 object-contain bg-gray-50"
                      />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Processing Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5" />
                  <span>Processing Configuration</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="provider">AI Provider</Label>
                  <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((provider) => (
                        <SelectItem key={provider.provider} value={provider.provider}>
                          <div className="flex items-center space-x-2">
                            <span>{provider.displayName}</span>
                            {provider.isAvailable && (
                              <Badge variant="secondary" className="text-xs">Available</Badge>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="max-results">Max Results</Label>
                    <Input
                      id="max-results"
                      type="number"
                      min="1"
                      max="50"
                      value={maxResults}
                      onChange={(e) => setMaxResults(parseInt(e.target.value) || 10)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="min-confidence">Min Confidence (%)</Label>
                    <Input
                      id="min-confidence"
                      type="number"
                      min="0"
                      max="100"
                      value={minConfidence}
                      onChange={(e) => setMinConfidence(parseInt(e.target.value) || 50)}
                      className="mt-1"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Processing Types */}
          <Card>
            <CardHeader>
              <CardTitle>Processing Types</CardTitle>
              <p className="text-sm text-muted-foreground">
                Select the types of analysis to perform on your image
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {PROCESSING_TYPES.map((type) => {
                  const Icon = type.icon;
                  const isAvailable = availableTypes.includes(type.value);
                  const isSelected = selectedTypes.includes(type.value);
                  
                  return (
                    <div
                      key={type.value}
                      className={`flex items-start space-x-3 p-3 rounded-lg border ${
                        isAvailable 
                          ? 'border-border hover:border-primary/50 cursor-pointer' 
                          : 'border-muted bg-muted/50 cursor-not-allowed opacity-50'
                      } ${isSelected ? 'border-primary bg-primary/5' : ''}`}
                      onClick={() => isAvailable && handleProcessingTypeToggle(type.value, !isSelected)}
                    >
                      <Checkbox
                        checked={isSelected}
                        disabled={!isAvailable}
                        onChange={(checked) => handleProcessingTypeToggle(type.value, checked)}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <Icon className="h-4 w-4 flex-shrink-0" />
                          <span className="font-medium text-sm">{type.label}</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {type.description}
                        </p>
                        {!isAvailable && (
                          <Badge variant="outline" className="text-xs mt-1">
                            Not available for {selectedProviderInfo?.displayName}
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* Process Button */}
          <div className="flex justify-center">
            <Button
              onClick={processImage}
              disabled={processing || (!selectedFile && !imageUrl) || selectedTypes.length === 0}
              size="lg"
              className="px-8"
            >
              {processing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Brain className="h-4 w-4 mr-2" />
                  Process Image
                </>
              )}
            </Button>
          </div>

          {/* Current Job Status */}
          {currentJob && (
            <Card>
              <CardHeader>
                <CardTitle>Current Job Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span>Job ID: {currentJob.id}</span>
                    <Badge variant={
                      currentJob.status === 'COMPLETED' ? 'default' :
                      currentJob.status === 'FAILED' ? 'destructive' :
                      currentJob.status === 'PROCESSING' ? 'secondary' : 'outline'
                    }>
                      {currentJob.status}
                    </Badge>
                  </div>
                  
                  {currentJob.status === 'PROCESSING' && (
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Processing...</span>
                        <span>Using {currentJob.provider}</span>
                      </div>
                      <Progress value={undefined} className="w-full" />
                    </div>
                  )}
                  
                  {currentJob.errorMessage && (
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>{currentJob.errorMessage}</AlertDescription>
                    </Alert>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Processing History</CardTitle>
            </CardHeader>
            <CardContent>
              {jobHistory.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  No processing history yet. Process your first image to see results here.
                </div>
              ) : (
                <div className="space-y-4">
                  {jobHistory.map((job) => (
                    <div key={job.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="flex-shrink-0">
                          {React.createElement(getProcessingTypeIcon(job.processingType), { className: "h-5 w-5" })}
                        </div>
                        <div>
                          <div className="font-medium">{job.id}</div>
                          <div className="text-sm text-muted-foreground">
                            {new Date(job.createdAt).toLocaleString()}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <Badge variant={
                          job.status === 'COMPLETED' ? 'default' :
                          job.status === 'FAILED' ? 'destructive' :
                          'secondary'
                        }>
                          {job.status}
                        </Badge>
                        {job.costCents && (
                          <span className="text-sm text-muted-foreground">
                            {formatCost(job.costCents)}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              {currentJob?.results ? (
                <div className="space-y-4">
                  {/* Results would be displayed here based on the processing types */}
                  <div className="text-center py-8 text-muted-foreground">
                    Results visualization will be implemented based on the processing types selected.
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No results to display. Process an image to see analysis results here.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
