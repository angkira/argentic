import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { combineLatest, Subject } from 'rxjs';
import { map, startWith, catchError, switchMap, tap, shareReplay } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import {
  LLMProviderInfo,
  MessagingProtocolInfo,
  LLMProviderConfig,
  MessagingConfig
} from '../../models';

interface ConfigState {
  llmProviders: LLMProviderInfo[];
  messagingProtocols: MessagingProtocolInfo[];
  loading: boolean;
  error: string | null;
}

@Component({
  selector: 'app-config',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './config.component.html',
  styleUrls: ['./config.component.scss']
})
export class ConfigComponent {
  // Signals for local UI state
  llmSaveStatus = signal<'idle' | 'saving' | 'success' | 'error'>('idle');
  llmSaveMessage = signal<string>('');
  messagingSaveStatus = signal<'idle' | 'saving' | 'success' | 'error'>('idle');
  messagingSaveMessage = signal<string>('');

  // Action subjects
  private saveLLMAction$ = new Subject<LLMProviderConfig>();
  private saveMessagingAction$ = new Subject<MessagingConfig>();

  // Form data (mutable for two-way binding)
  llmConfig: LLMProviderConfig = {
    provider: 'google_gemini',
    google_gemini_api_key: '',
    google_gemini_model_name: 'gemini-2.0-flash',
    parameters: {
      temperature: 0.7,
      top_p: 0.95,
      max_output_tokens: 2048
    }
  };

  messagingConfig: MessagingConfig = {
    protocol: 'mqtt',
    broker_address: 'localhost',
    port: 1883,
    keepalive: 60,
    use_tls: false
  };

  // Declarative data stream for config options
  readonly state$ = combineLatest({
    llmProviders: this.apiService.getLLMProviders(),
    messagingProtocols: this.apiService.getMessagingProtocols()
  }).pipe(
    map(({ llmProviders, messagingProtocols }) => ({
      llmProviders,
      messagingProtocols,
      loading: false,
      error: null
    } as ConfigState)),
    startWith({
      llmProviders: [],
      messagingProtocols: [],
      loading: true,
      error: null
    } as ConfigState),
    catchError(error => {
      console.error('Error loading config options:', error);
      return [{
        llmProviders: [],
        messagingProtocols: [],
        loading: false,
        error: error.message
      } as ConfigState];
    }),
    shareReplay({ bufferSize: 1, refCount: true }),
    takeUntilDestroyed()
  );

  // LLM save effect
  private readonly saveLLMEffect$ = this.saveLLMAction$.pipe(
    tap(() => {
      this.llmSaveStatus.set('saving');
      this.llmSaveMessage.set('');
    }),
    switchMap(config =>
      this.apiService.validateLLMConfig(config).pipe(
        tap(() => {
          this.llmSaveStatus.set('success');
          this.llmSaveMessage.set('LLM configuration saved successfully!');
          setTimeout(() => this.llmSaveStatus.set('idle'), 3000);
        }),
        catchError(error => {
          this.llmSaveStatus.set('error');
          this.llmSaveMessage.set(`Error: ${error.error?.detail || error.message}`);
          return [];
        })
      )
    ),
    takeUntilDestroyed()
  );

  // Messaging save effect
  private readonly saveMessagingEffect$ = this.saveMessagingAction$.pipe(
    tap(() => {
      this.messagingSaveStatus.set('saving');
      this.messagingSaveMessage.set('');
    }),
    switchMap(config =>
      this.apiService.validateMessagingConfig(config).pipe(
        tap(() => {
          this.messagingSaveStatus.set('success');
          this.messagingSaveMessage.set('Messaging configuration saved successfully!');
          setTimeout(() => this.messagingSaveStatus.set('idle'), 3000);
        }),
        catchError(error => {
          this.messagingSaveStatus.set('error');
          this.messagingSaveMessage.set(`Error: ${error.error?.detail || error.message}`);
          return [];
        })
      )
    ),
    takeUntilDestroyed()
  );

  constructor(private apiService: ApiService) {
    // Subscribe to effects to activate them
    this.saveLLMEffect$.subscribe();
    this.saveMessagingEffect$.subscribe();
  }

  saveLLMConfig(): void {
    this.saveLLMAction$.next(this.llmConfig);
  }

  saveMessagingConfig(): void {
    this.saveMessagingAction$.next(this.messagingConfig);
  }

  onProviderChange(): void {
    // Reset provider-specific fields when changing provider
  }
}
