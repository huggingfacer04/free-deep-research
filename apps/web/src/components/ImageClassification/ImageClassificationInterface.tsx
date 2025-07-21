import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Brain, 
  Upload, 
  Play, 
  Pause, 
  Settings, 
  BarChart3, 
  Layers, 
  Zap,
  CheckCircle,
  AlertCircle,
  Clock,
  TrendingUp
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Types for image classification
interface ImageClassificationModel {
  id: string;
  name: string;
  modelType: string;
  version: string;
  framework: string;
  status: 'DRAFT' | 'TRAINING' | 'TRAINED' | 'DEPLOYED' | 'FAILED';
  accuracyMetrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    validationLoss: number;
    trainingLoss: number;
    inferenceTimeMs: number;
  };
  createdAt: string;
  updatedAt: string;
}

interface TrainingJob {
  jobId: string;
  modelId: string;
  jobName: string;
  status: 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED';
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  currentMetrics?: {
    trainingLoss: number;
    trainingAccuracy: number;
    validationLoss: number;
    validationAccuracy: number;
    learningRate: number;
  };
  startedAt: string;
  estimatedCompletion?: string;
  errorMessage?: string;
}

interface TrainingStatistics {
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  activeJobs: number;
  totalModels: number;
  deployedModels: number;
}

const MODEL_TYPES = [
  { value: 'GENERAL_CLASSIFICATION', label: 'General Classification', description: 'Multi-purpose image classification' },
  { value: 'DOCUMENT_CLASSIFICATION', label: 'Document Classification', description: 'Research papers, reports, documents' },
  { value: 'MEDICAL_IMAGE_CLASSIFICATION', label: 'Medical Images', description: 'Medical imaging and diagnostics' },
  { value: 'SCIENTIFIC_DIAGRAM_CLASSIFICATION', label: 'Scientific Diagrams', description: 'Charts, graphs, scientific figures' },
  { value: 'IMAGE_QUALITY_ASSESSMENT', label: 'Quality Assessment', description: 'Image quality and aesthetics' },
  { value: 'CONTENT_MODERATION_CLASSIFICATION', label: 'Content Moderation', description: 'Safety and content filtering' },
];

const FRAMEWORKS = [
  { value: 'TENSORFLOW', label: 'TensorFlow', description: 'Google\'s ML framework' },
  { value: 'PYTORCH', label: 'PyTorch', description: 'Facebook\'s ML framework' },
  { value: 'KERAS', label: 'Keras', description: 'High-level neural networks API' },
];

const ARCHITECTURES = [
  { value: 'CNN', label: 'Custom CNN', description: 'Custom Convolutional Neural Network' },
  { value: 'RESNET50', label: 'ResNet-50', description: 'Pre-trained ResNet-50 architecture' },
  { value: 'EFFICIENTNET_B0', label: 'EfficientNet-B0', description: 'Efficient and accurate CNN' },
  { value: 'VISION_TRANSFORMER', label: 'Vision Transformer', description: 'Transformer-based architecture' },
];

export const ImageClassificationInterface: React.FC = () => {
  const [models, setModels] = useState<ImageClassificationModel[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [statistics, setStatistics] = useState<TrainingStatistics | null>(null);
  const [selectedModel, setSelectedModel] = useState<ImageClassificationModel | null>(null);
  const [activeTab, setActiveTab] = useState('models');
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  // Model creation form state
  const [newModelName, setNewModelName] = useState('');
  const [newModelType, setNewModelType] = useState('');
  const [newModelFramework, setNewModelFramework] = useState('');
  const [newModelArchitecture, setNewModelArchitecture] = useState('');

  // Training configuration state
  const [datasetPath, setDatasetPath] = useState('');
  const [numClasses, setNumClasses] = useState(10);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);

  useEffect(() => {
    loadModels();
    loadTrainingJobs();
    loadStatistics();
  }, []);

  const loadModels = async () => {
    try {
      // This would call the Tauri command
      // const models = await invoke('list_user_image_classification_models', { userId: 'current-user' });
      // For now, mock data
      const mockModels: ImageClassificationModel[] = [
        {
          id: 'model-1',
          name: 'Research Paper Classifier',
          modelType: 'DOCUMENT_CLASSIFICATION',
          version: '1.0.0',
          framework: 'TENSORFLOW',
          status: 'TRAINED',
          accuracyMetrics: {
            accuracy: 0.94,
            precision: 0.93,
            recall: 0.95,
            f1Score: 0.94,
            validationLoss: 0.15,
            trainingLoss: 0.12,
            inferenceTimeMs: 45,
          },
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      ];
      setModels(mockModels);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load models',
        variant: 'destructive',
      });
    }
  };

  const loadTrainingJobs = async () => {
    try {
      // Mock training jobs
      const mockJobs: TrainingJob[] = [];
      setTrainingJobs(mockJobs);
    } catch (error) {
      console.error('Failed to load training jobs:', error);
    }
  };

  const loadStatistics = async () => {
    try {
      // This would call the Tauri command
      // const stats = await invoke('get_image_classification_statistics');
      const mockStats: TrainingStatistics = {
        totalJobs: 5,
        completedJobs: 3,
        failedJobs: 1,
        activeJobs: 1,
        totalModels: 4,
        deployedModels: 2,
      };
      setStatistics(mockStats);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    }
  };

  const createModel = async () => {
    if (!newModelName || !newModelType || !newModelFramework || !newModelArchitecture) {
      toast({
        title: 'Validation Error',
        description: 'Please fill in all required fields',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);
    try {
      // This would call the Tauri command
      // const modelId = await invoke('create_image_classification_model', {
      //   name: newModelName,
      //   modelType: newModelType,
      //   framework: newModelFramework,
      //   architecture: { /* architecture config */ },
      //   createdBy: 'current-user'
      // });

      toast({
        title: 'Model Created',
        description: `Successfully created model: ${newModelName}`,
      });

      // Reset form
      setNewModelName('');
      setNewModelType('');
      setNewModelFramework('');
      setNewModelArchitecture('');

      // Reload models
      loadModels();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to create model',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const startTraining = async (modelId: string) => {
    if (!datasetPath) {
      toast({
        title: 'Validation Error',
        description: 'Please specify a dataset path',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);
    try {
      // This would call the Tauri command
      // const jobId = await invoke('start_image_classification_training', {
      //   modelId,
      //   trainingConfig: {
      //     datasetConfig: {
      //       datasetPath,
      //       numClasses,
      //       // ... other config
      //     },
      //     trainingParams: {
      //       epochs,
      //       batchSize,
      //       learningRate,
      //       // ... other params
      //     }
      //   },
      //   userId: 'current-user'
      // });

      toast({
        title: 'Training Started',
        description: 'Model training has been initiated',
      });

      loadTrainingJobs();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to start training',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const deployModel = async (modelId: string) => {
    setLoading(true);
    try {
      // This would call the Tauri command
      // const endpoint = await invoke('deploy_image_classification_model', {
      //   modelId,
      //   deploymentConfig: {
      //     // deployment configuration
      //   }
      // });

      toast({
        title: 'Model Deployed',
        description: 'Model has been successfully deployed',
      });

      loadModels();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to deploy model',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'TRAINED':
      case 'DEPLOYED':
      case 'COMPLETED':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'TRAINING':
        return <Clock className="h-4 w-4 text-blue-500" />;
      case 'FAILED':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatAccuracy = (accuracy: number) => {
    return `${(accuracy * 100).toFixed(1)}%`;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Image Classification</h1>
          <p className="text-muted-foreground">
            Train and deploy custom image classification models
          </p>
        </div>
        {statistics && (
          <div className="grid grid-cols-3 gap-4">
            <Card className="p-4">
              <div className="flex items-center space-x-2">
                <Brain className="h-4 w-4 text-blue-500" />
                <div className="text-sm">
                  <div className="font-medium">{statistics.totalModels}</div>
                  <div className="text-muted-foreground">Models</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-4 w-4 text-green-500" />
                <div className="text-sm">
                  <div className="font-medium">{statistics.deployedModels}</div>
                  <div className="text-muted-foreground">Deployed</div>
                </div>
              </div>
            </Card>
            <Card className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-purple-500" />
                <div className="text-sm">
                  <div className="font-medium">{statistics.activeJobs}</div>
                  <div className="text-muted-foreground">Training</div>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="create">Create Model</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="inference">Inference</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {models.map((model) => (
              <Card key={model.id} className="cursor-pointer hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{model.name}</CardTitle>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(model.status)}
                      <Badge variant={model.status === 'DEPLOYED' ? 'default' : 'secondary'}>
                        {model.status}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {MODEL_TYPES.find(t => t.value === model.modelType)?.label}
                  </p>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Framework:</span>
                      <div className="font-medium">{model.framework}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Version:</span>
                      <div className="font-medium">{model.version}</div>
                    </div>
                  </div>

                  {model.accuracyMetrics && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Accuracy:</span>
                        <span className="font-medium">{formatAccuracy(model.accuracyMetrics.accuracy)}</span>
                      </div>
                      <Progress value={model.accuracyMetrics.accuracy * 100} className="h-2" />
                      
                      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                        <div>Precision: {formatAccuracy(model.accuracyMetrics.precision)}</div>
                        <div>Recall: {formatAccuracy(model.accuracyMetrics.recall)}</div>
                        <div>F1-Score: {formatAccuracy(model.accuracyMetrics.f1Score)}</div>
                        <div>Inference: {model.accuracyMetrics.inferenceTimeMs}ms</div>
                      </div>
                    </div>
                  )}

                  <div className="flex space-x-2">
                    {model.status === 'TRAINED' && (
                      <Button
                        size="sm"
                        onClick={() => deployModel(model.id)}
                        disabled={loading}
                      >
                        <Zap className="h-3 w-3 mr-1" />
                        Deploy
                      </Button>
                    )}
                    {model.status === 'DRAFT' && (
                      <Button
                        size="sm"
                        onClick={() => {
                          setSelectedModel(model);
                          setActiveTab('training');
                        }}
                        disabled={loading}
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Train
                      </Button>
                    )}
                    <Button size="sm" variant="outline">
                      <Settings className="h-3 w-3 mr-1" />
                      Configure
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="create" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create New Model</CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure a new image classification model
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="model-name">Model Name</Label>
                    <Input
                      id="model-name"
                      placeholder="Enter model name"
                      value={newModelName}
                      onChange={(e) => setNewModelName(e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="model-type">Model Type</Label>
                    <Select value={newModelType} onValueChange={setNewModelType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select model type" />
                      </SelectTrigger>
                      <SelectContent>
                        {MODEL_TYPES.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            <div>
                              <div className="font-medium">{type.label}</div>
                              <div className="text-xs text-muted-foreground">{type.description}</div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="framework">Framework</Label>
                    <Select value={newModelFramework} onValueChange={setNewModelFramework}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select framework" />
                      </SelectTrigger>
                      <SelectContent>
                        {FRAMEWORKS.map((framework) => (
                          <SelectItem key={framework.value} value={framework.value}>
                            <div>
                              <div className="font-medium">{framework.label}</div>
                              <div className="text-xs text-muted-foreground">{framework.description}</div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="architecture">Architecture</Label>
                    <Select value={newModelArchitecture} onValueChange={setNewModelArchitecture}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select architecture" />
                      </SelectTrigger>
                      <SelectContent>
                        {ARCHITECTURES.map((arch) => (
                          <SelectItem key={arch.value} value={arch.value}>
                            <div>
                              <div className="font-medium">{arch.label}</div>
                              <div className="text-xs text-muted-foreground">{arch.description}</div>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <Alert>
                    <Layers className="h-4 w-4" />
                    <AlertDescription>
                      The model architecture will be automatically configured based on your selection.
                      You can customize it further after creation.
                    </AlertDescription>
                  </Alert>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={createModel} disabled={loading}>
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Create Model
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="training" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure training parameters for your model
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="dataset-path">Dataset Path</Label>
                    <Input
                      id="dataset-path"
                      placeholder="/datasets/my-dataset"
                      value={datasetPath}
                      onChange={(e) => setDatasetPath(e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="num-classes">Number of Classes</Label>
                    <Input
                      id="num-classes"
                      type="number"
                      min="2"
                      max="1000"
                      value={numClasses}
                      onChange={(e) => setNumClasses(parseInt(e.target.value) || 10)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input
                      id="epochs"
                      type="number"
                      min="1"
                      max="1000"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="batch-size">Batch Size</Label>
                    <Input
                      id="batch-size"
                      type="number"
                      min="1"
                      max="512"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="learning-rate">Learning Rate</Label>
                    <Input
                      id="learning-rate"
                      type="number"
                      step="0.0001"
                      min="0.0001"
                      max="1"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                    />
                  </div>

                  {selectedModel && (
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Selected Model</h4>
                      <p className="text-sm text-muted-foreground">{selectedModel.name}</p>
                      <Badge variant="outline" className="mt-2">
                        {selectedModel.framework}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex justify-end">
                <Button
                  onClick={() => selectedModel && startTraining(selectedModel.id)}
                  disabled={loading || !selectedModel}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start Training
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Training Jobs */}
          {trainingJobs.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Active Training Jobs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {trainingJobs.map((job) => (
                    <div key={job.jobId} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{job.jobName}</h4>
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(job.status)}
                          <Badge variant="secondary">{job.status}</Badge>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress:</span>
                          <span>{job.progress.toFixed(1)}%</span>
                        </div>
                        <Progress value={job.progress} className="h-2" />
                        
                        <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
                          <div>Epoch: {job.currentEpoch}/{job.totalEpochs}</div>
                          <div>Started: {new Date(job.startedAt).toLocaleString()}</div>
                        </div>
                        
                        {job.currentMetrics && (
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>Training Acc: {formatAccuracy(job.currentMetrics.trainingAccuracy)}</div>
                            <div>Val Acc: {formatAccuracy(job.currentMetrics.validationAccuracy)}</div>
                            <div>Training Loss: {job.currentMetrics.trainingLoss.toFixed(4)}</div>
                            <div>Val Loss: {job.currentMetrics.validationLoss.toFixed(4)}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="inference" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Inference</CardTitle>
              <p className="text-sm text-muted-foreground">
                Test your deployed models with sample images
              </p>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Upload className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Inference interface will be available once models are deployed.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
