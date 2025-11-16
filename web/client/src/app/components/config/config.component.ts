import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  LLMProviderInfo,
  MessagingProtocolInfo,
  LLMProviderConfig,
  MessagingConfig
} from '../../models';

@Component({
  selector: 'app-config',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './config.component.html',
  styleUrls: ['./config.component.scss']
})
export class ConfigComponent implements OnInit {
  llmProviders: LLMProviderInfo[] = [];
  messagingProtocols: MessagingProtocolInfo[] = [];
  loading = true;

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

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadConfigOptions();
  }

  loadConfigOptions(): void {
    this.loading = true;
    Promise.all([
      this.apiService.getLLMProviders().toPromise(),
      this.apiService.getMessagingProtocols().toPromise()
    ]).then(([llmProviders, messagingProtocols]) => {
      this.llmProviders = llmProviders || [];
      this.messagingProtocols = messagingProtocols || [];
      this.loading = false;
    }).catch(error => {
      console.error('Error loading config options:', error);
      this.loading = false;
    });
  }

  saveLLMConfig(): void {
    this.apiService.validateLLMConfig(this.llmConfig).subscribe({
      next: () => alert('LLM configuration saved!'),
      error: (error) => alert(`Error saving LLM config: ${error.error?.detail || error.message}`)
    });
  }

  saveMessagingConfig(): void {
    this.apiService.validateMessagingConfig(this.messagingConfig).subscribe({
      next: () => alert('Messaging configuration saved!'),
      error: (error) => alert(`Error saving messaging config: ${error.error?.detail || error.message}`)
    });
  }

  onProviderChange(): void {
    // Reset provider-specific fields when changing provider
  }
}
