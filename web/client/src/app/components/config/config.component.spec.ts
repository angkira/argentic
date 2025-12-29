import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { ConfigComponent } from './config.component';
import { ApiService } from '../../services/api.service';
import { of, throwError } from 'rxjs';
import { LLMProviderInfo, MessagingProtocolInfo, LLMProviderConfig, MessagingConfig } from '../../models';

describe('ConfigComponent', () => {
  let component: ConfigComponent;
  let fixture: ComponentFixture<ConfigComponent>;
  let apiService: jasmine.SpyObj<ApiService>;

  const mockLLMProviders: LLMProviderInfo[] = [
    {
      name: 'google_gemini',
      display_name: 'Google Gemini',
      description: 'Google Gemini API',
      supported_models: ['gemini-2.0-flash'],
      required_credentials: ['api_key']
    },
    {
      name: 'openai',
      display_name: 'OpenAI',
      description: 'OpenAI API',
      supported_models: ['gpt-4', 'gpt-3.5-turbo'],
      required_credentials: ['api_key']
    }
  ];

  const mockMessagingProtocols: MessagingProtocolInfo[] = [
    {
      name: 'mqtt',
      display_name: 'MQTT',
      description: 'MQTT Protocol',
      default_port: 1883,
      supports_tls: true
    },
    {
      name: 'redis',
      display_name: 'Redis',
      description: 'Redis Pub/Sub',
      default_port: 6379,
      supports_tls: true
    }
  ];

  beforeEach(async () => {
    const apiServiceSpy = jasmine.createSpyObj('ApiService', [
      'getLLMProviders',
      'getMessagingProtocols',
      'validateLLMConfig',
      'validateMessagingConfig'
    ]);

    await TestBed.configureTestingModule({
      imports: [ConfigComponent, HttpClientTestingModule, FormsModule],
      providers: [
        { provide: ApiService, useValue: apiServiceSpy }
      ]
    }).compileComponents();

    apiService = TestBed.inject(ApiService) as jasmine.SpyObj<ApiService>;
    apiService.getLLMProviders.and.returnValue(of(mockLLMProviders));
    apiService.getMessagingProtocols.and.returnValue(of(mockMessagingProtocols));

    fixture = TestBed.createComponent(ConfigComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have state$ observable that emits initial loading state', (done) => {
    component.state$.subscribe(state => {
      if (state.loading) {
        expect(state.llmProviders).toEqual([]);
        expect(state.messagingProtocols).toEqual([]);
        expect(state.loading).toBe(true);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should emit config data after loading', (done) => {
    let emissionCount = 0;
    component.state$.subscribe(state => {
      emissionCount++;
      if (emissionCount === 2 && !state.loading) {
        expect(state.llmProviders).toEqual(mockLLMProviders);
        expect(state.messagingProtocols).toEqual(mockMessagingProtocols);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
        done();
      }
    });
  });

  it('should initialize with default LLM config', () => {
    expect(component.llmConfig.provider).toBe('google_gemini');
    expect(component.llmConfig.google_gemini_model_name).toBe('gemini-2.0-flash');
    expect(component.llmConfig.parameters?.temperature).toBe(0.7);
  });

  it('should initialize with default messaging config', () => {
    expect(component.messagingConfig.protocol).toBe('mqtt');
    expect(component.messagingConfig.broker_address).toBe('localhost');
    expect(component.messagingConfig.port).toBe(1883);
    expect(component.messagingConfig.use_tls).toBe(false);
  });

  it('should initialize save status signals to idle', () => {
    expect(component.llmSaveStatus()).toBe('idle');
    expect(component.llmSaveMessage()).toBe('');
    expect(component.messagingSaveStatus()).toBe('idle');
    expect(component.messagingSaveMessage()).toBe('');
  });

  it('should save LLM config successfully', fakeAsync(() => {
    apiService.validateLLMConfig.and.returnValue(of({ valid: true, provider: 'google_gemini' } as any));

    component.saveLLMConfig();

    expect(component.llmSaveStatus()).toBe('saving');

    tick(50);

    expect(apiService.validateLLMConfig).toHaveBeenCalledWith(component.llmConfig);
    expect(component.llmSaveStatus()).toBe('success');
    expect(component.llmSaveMessage()).toBe('LLM configuration saved successfully!');

    tick(3000);

    expect(component.llmSaveStatus()).toBe('idle');
  }));

  it('should handle LLM config save error', fakeAsync(() => {
    const errorResponse = { error: { detail: 'Invalid API key' }, message: 'Validation failed' };
    apiService.validateLLMConfig.and.returnValue(
      throwError(() => errorResponse)
    );

    component.saveLLMConfig();

    expect(component.llmSaveStatus()).toBe('saving');

    tick(50);

    expect(component.llmSaveStatus()).toBe('error');
    expect(component.llmSaveMessage()).toBe('Error: Invalid API key');
  }));

  it('should use error message if detail not available', fakeAsync(() => {
    const errorResponse = { message: 'Network error' };
    apiService.validateLLMConfig.and.returnValue(
      throwError(() => errorResponse)
    );

    component.saveLLMConfig();

    tick(50);

    expect(component.llmSaveStatus()).toBe('error');
    expect(component.llmSaveMessage()).toBe('Error: Network error');
  }));

  it('should save messaging config successfully', fakeAsync(() => {
    apiService.validateMessagingConfig.and.returnValue(of({ valid: true, provider: 'google_gemini' } as any));

    component.saveMessagingConfig();

    expect(component.messagingSaveStatus()).toBe('saving');

    tick(50);

    expect(apiService.validateMessagingConfig).toHaveBeenCalledWith(component.messagingConfig);
    expect(component.messagingSaveStatus()).toBe('success');
    expect(component.messagingSaveMessage()).toBe('Messaging configuration saved successfully!');

    tick(3000);

    expect(component.messagingSaveStatus()).toBe('idle');
  }));

  it('should handle messaging config save error', fakeAsync(() => {
    const errorResponse = { error: { detail: 'Connection failed' }, message: 'Validation failed' };
    apiService.validateMessagingConfig.and.returnValue(
      throwError(() => errorResponse)
    );

    component.saveMessagingConfig();

    expect(component.messagingSaveStatus()).toBe('saving');

    tick(50);

    expect(component.messagingSaveStatus()).toBe('error');
    expect(component.messagingSaveMessage()).toBe('Error: Connection failed');
  }));

  it('should handle API error when loading config', (done) => {
    apiService.getLLMProviders.and.returnValue(
      throwError(() => ({ message: 'Failed to load providers' }))
    );

    const newComponent = new ConfigComponent(apiService);

    newComponent.state$.subscribe(state => {
      if (state.error) {
        expect(state.loading).toBe(false);
        expect(state.error).toBe('Failed to load providers');
        expect(state.llmProviders).toEqual([]);
        expect(state.messagingProtocols).toEqual([]);
        done();
      }
    });
  });

  it('should clear save message when saving new config', fakeAsync(() => {
    component.llmSaveMessage.set('Previous message');
    apiService.validateLLMConfig.and.returnValue(of({ valid: true, provider: 'google_gemini' } as any));

    component.saveLLMConfig();

    expect(component.llmSaveMessage()).toBe('');
  }));

  it('should update save status through complete flow', fakeAsync(() => {
    apiService.validateLLMConfig.and.returnValue(of({ valid: true, provider: 'google_gemini' } as any));

    expect(component.llmSaveStatus()).toBe('idle');

    component.saveLLMConfig();
    expect(component.llmSaveStatus()).toBe('saving');

    tick(50);
    expect(component.llmSaveStatus()).toBe('success');

    tick(3000);
    expect(component.llmSaveStatus()).toBe('idle');
  }));

  it('should have onProviderChange method', () => {
    expect(component.onProviderChange).toBeDefined();
    component.onProviderChange();
    // Method exists but doesn't do anything currently
  });
});
